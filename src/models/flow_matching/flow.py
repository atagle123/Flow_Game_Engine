import os
import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization
from flax import struct
from flax.training.train_state import TrainState
from functools import partial
from src.jaxrl5.types import PRNGKey
from functools import partial
import jax
import jax.numpy as jnp
import gym
import optax
from flax import struct
from flax.training.train_state import TrainState
from typing import Dict, Tuple
import flax.linen as nn
from src.jaxrl5.data.dataset import DatasetDict
from src.models.critic_model import Ensemble, Q_Model, ValueModel
from src.models.diffusion_model import DDPM, ddpm_sampler, ddpm_sampler_swg
from src.models.models import MLP, MLPResNet
from src.models.helpers import cosine_beta_schedule, vp_beta_schedule, linear_beta_schedule, FourierFeatures
from src.models.weights import build_weights_fn, expectile_loss, quantile_loss, exponential_loss

def count_params(params): 
    return sum(jnp.prod(jnp.array(v.shape)) for v in jax.tree_util.tree_leaves(params))

class FlowLearner(struct.PyTreeNode):
    flow_model: TrainState
    target_flow_model: TrainState
    flow_tau: float
    act_dim: int = struct.field(pytree_node=False)
    n_timesteps: int = struct.field(pytree_node=False)
    rng: PRNGKey

    # data dim, t steps, interpolant, 

    @classmethod
    def create(
        cls,
        seed: int,
        cfg):

        rng = jax.random.PRNGKey(seed)
        rng, flow_key = jax.random.split(rng, 4)

        ### Flow model ###

        time_embedding_cls = partial(FourierFeatures,
                                      output_size=cfg.diffusion.time_emb,
                                      learnable=True)

        conditional_model_cls = partial(MLP,
                                hidden_dims=(cfg.diffusion.time_emb * 2, cfg.diffusion.time_emb * 2),
                                activations=nn.swish,
                                activate_final=False)

        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=cfg.diffusion.use_layer_norm,
                                    num_blocks=cfg.diffusion.num_blocks,
                                    dropout_rate=cfg.diffusion.dropout_rate,
                                    hidden_dim=cfg.diffusion.hidden_dim, 
                                    out_dim=action_dim+1,
                                    activations=nn.swish)
                                    
        
        flow_model = FlowModel(time_preprocess_cls=time_embedding_cls,
                            cond_encoder_cls=conditional_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        actions = jnp.expand_dims(actions, axis = 0)

        flow_params = flow_model.init(flow_key, observations, actions, time)['params']

        print(f"PARAMS: {count_params(flow_params)}")
        
        if cfg.train.cosine_decay:
            lr = optax.cosine_decay_schedule(cfg.train.lr, int(cfg.train.actor_steps))

        flow_model = TrainState.create(apply_fn=flow_model.apply,
                                        params=flow_params,
                                        tx=optax.adamw(learning_rate=actor_lr)) 
        
        target_flow_model = TrainState.create(apply_fn=flow_model.apply,
                                               params=flow_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        schedulers = {"vp": vp_beta_schedule,
                    "cosine": cosine_beta_schedule,
                    "linear": linear_beta_schedule}
        
        if cfg.diffusion.schedule in schedulers:
            betas = jnp.array(schedulers[cfg.diffusion.schedule](cfg.diffusion.n_timesteps))
        else:
            raise ValueError(f'Invalid beta schedule: {cfg.diffusion.schedule}')

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            rng=rng,
            act_dim=action_dim,
            n_timesteps=cfg.diffusion.n_timesteps,
            flow_tau=cfg.train.ema_update, )

    def update(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        time = jax.random.randint(key, (batch['action_weight'].shape[0], ), 0, agent.n_timesteps) # 0 TO? TODO sample continuous

        key, rng = jax.random.split(rng, 2)
        #noise_sample = jax.random.normal(key, (batch['action_weight'].shape[0], agent.act_dim+1)) # B,A+1
        
        #alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        intermediate_data = interpolate(x0, x1, t, key) # TODO
        velocity = #...
        key, rng = jax.random.split(rng, 2)

        def flow_loss_fn(flow_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vel_pred = agent.flow_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions_weights,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            loss = ((vel_pred - noise_sample) ** 2) # B,A+1
            loss = jnp.mean(jnp.sum(loss, axis=-1))
            return loss, {'loss': loss}

        grads, info = jax.grad(flow_loss_fn, has_aux=True)(agent.flow_model.params)
        flow_model = agent.flow_model.apply_gradients(grads=grads)

        agent = agent.replace(flow_model=flow_model)
        target_flow_params = optax.incremental_update(flow_model.params, agent.target_flow_model.params, agent.tau)

        target_flow_model = agent.target_flow_model.replace(params=target_flow_params)
        new_agent = agent.replace(flow_model=flow_model, target_flow_model=target_flow_model, rng=rng)

        return new_agent, info


    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update(batch)
        return new_agent, actor_info
    

    def sample(self, observations: jnp.ndarray, rng, sample_params):
        #rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(sample_params["batch_size"], axis = 0)

        score_params = self.target_score_model.params
        actions_weights, rng = ddpm_sampler_swg(self.score_model.apply_fn, 
                                    score_params, 
                                    self.n_timesteps, 
                                    rng, 
                                    self.act_dim+1, 
                                    observations,
                                    self.alphas, 
                                    self.alpha_hats, 
                                    self.betas,
                                    temperature=sample_params["temperature"], 
                                    repeat_last_step=0, 
                                    clip_denoised=sample_params["clip_denoised"], 
                                    guidance_scale=sample_params["guidance_scale"], 
                                    max_weight_clip=sample_params["max_weight_clip"], 
                                    min_weight_clip=sample_params["min_weight_clip"])
        
       # actions_weights, rng =ddpm_sampler(self.score_model.apply_fn, score_params, self.n_timesteps, rng, self.act_dim+1, observations, self.alphas, self.alpha_hats, self.betas, temperature=sample_params["temperature"], repeat_last_step=0, clip_sampler=sample_params["clip_denoised"])

        new_rng, _ = jax.random.split(rng, 2)
        actions = actions_weights[:, :-1]
        weights = actions_weights[:, -1]
        #action=actions
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions) 
        idx = jnp.argmax(qs)
        action = actions[idx]
        return action, new_rng


    def save(self, savepath, step):

        actor_path = os.path.join(savepath, f"flow_params_{step}.msgpack")
        with open(actor_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.target_score_model.params))

    @classmethod
    def load(cls, cfg, dir: str, step): 
        """Load the parameters of the critic, value, and actor models."""
        # Create a new instance of the learner
        agent = cls.create(cfg=cfg.agent)

        # Load actor (score model) parameters
        actor_path = os.path.join(actor_dir, f"actor_params_{actor_step}.msgpack")
        with open(actor_path, "rb") as f:
            actor_params = flax.serialization.from_bytes(agent.target_score_model.params, f.read())
        agent = agent.replace(target_score_model=agent.target_score_model.replace(params=actor_params))

        print(f"Models loaded from {actor_dir} and {critic_dir}")
        return agent