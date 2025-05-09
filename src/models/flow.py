import os
import jax
import jax.numpy as jnp
import flax.serialization
from flax import struct
from flax.training.train_state import TrainState
from functools import partial
from typing import Dict, Tuple, Any
from jaxtyping import Array, Float
import optax
import flax.linen as nn
from models.flow_net import FlowModel
from models.helpers import CNN, FourierFeatures, MLP


PRNGKey = Any

def interpolant(x0: Float[Array, "N"] , x1: Float[Array, "N"], t: float) -> Float[Array, "N"]:
    return x0 + (x1 - x0) * t

velocity = jax.jacrev(interpolant, argnums=2)

interpolant = jax.vmap(interpolant)
velocity = jax.vmap(velocity)

def count_params(params): 
    return sum(jnp.prod(jnp.array(v.shape)) for v in jax.tree_util.tree_leaves(params))

class FlowLearner(struct.PyTreeNode): # TODO, training and sampling, interpolation func... 
    flow_model: TrainState
    target_flow_model: TrainState
    flow_tau: float
    rng: PRNGKey

    # data dim, t steps, interpolant, 

    @classmethod
    def create(cls,
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

        base_model_cls = partial(CNN) # add dim to encode conditions... TODO
                                    
        
        flow_model = FlowModel(time_preprocess_cls=time_embedding_cls, # check that the states are pas
                            cond_encoder_cls=conditional_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        x_dummy = jnp.zeros((1, 12, 12, 3)) # CHECK DIMS... TODO
        actions = jnp.zeros((1, 1))

        flow_params = flow_model.init(flow_key, x_dummy, actions, time)['params'] # TODO check init params

        print(f"PARAMS: {count_params(flow_params)}")
        
        if cfg.train.cosine_decay:
            lr = optax.cosine_decay_schedule(cfg.train.lr, int(cfg.train.steps))
        else: 
            lr = cfg.train.lr

        flow_model = TrainState.create(apply_fn=flow_model.apply,
                                        params=flow_params,
                                        tx=optax.adamw(learning_rate=lr)) 
        
        target_flow_model = TrainState.create(apply_fn=flow_model.apply,
                                               params=flow_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))        

        return cls(
                flow_model=flow_model,
                target_flow_model=target_flow_model,
                rng=rng,
                flow_tau=cfg.train.ema_update)

    def update_model(model, batch: DatasetDict) -> Tuple[FlowLearner, Dict[str, float]]:
        rng = model.rng
        key, rng = jax.random.split(rng, 2)

        x0 = batch["states"]
        x1 = batch["next_states"]

        t = jax.random.uniform(key, (x1.shape[0]))
        key, rng = jax.random.split(rng, 2)

        xt = interpolant(x0, x1, t)
        vt = velocity(x0, x1, t)

        def flow_loss_fn(flow_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vt_pred = model.flow_model.apply_fn({'params': flow_model_params},
                                       xt,
                                       batch["actions"],
                                       t,
                                       rngs={'dropout': key},
                                       training=True)
            
            loss = ((vt_pred - vt) ** 2)
            loss = jnp.mean(jnp.sum(loss, axis=-1))  #    loss = jnp.mean((vt_pred - vt) ** 2)
            return loss, {'loss': loss}

        grads, info = jax.grad(flow_loss_fn, has_aux=True)(model.flow_model.params)
        flow_model = model.flow_model.apply_gradients(grads=grads)

        model = model.replace(flow_model=flow_model)
        target_flow_params = optax.incremental_update(flow_model.params, model.target_flow_model.params, model.flow_tau)

        target_flow_model = model.target_flow_model.replace(params=target_flow_params)
        new_model = model.replace(flow_model=flow_model, target_flow_model=target_flow_model, rng=rng)

        return new_model, info


    @jax.jit
    def update(self, batch: DatasetDict):
        new_model = self
        new_model, info = new_model.update_model(batch)
        return new_model, info

    def sample(self, observation: jnp.ndarray, action: jnp.ndarray, rng, n_steps = 100):

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        actions = jax.device_put(actions)

        flow_params = self.target_flow_model.params
        timesteps = jnp.linspace(0, 1, n_steps)

        def flow_step(xt, t_pair):
            t_current, t_next = t_pair
            t = jnp.ones(1) * t_current
            dxt = self.flow_model.apply_fn(flow_params, t, xt, action)
            xt_new = xt + (t_next - t_current) * dxt
            return xt_new, None
        
        @jax.jit
        def run_flow(xt_init):
            xt_final, _ = jax.lax.scan(flow_step, xt_init, (timesteps[:-1], timesteps[1:]))
            return xt_final
        
        xt_final = run_flow(observation)
        
        new_rng, _ = jax.random.split(rng, 2)
        return xt_final, new_rng


    def save(self, savepath, step):

        actor_path = os.path.join(savepath, f"flow_params_{step}.msgpack")
        with open(actor_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.target_flow_model.params))

    @classmethod
    def load(cls, cfg, dir: str, step): 
        """Load the parameters of the flow model"""
        # Create a new instance of the learner
        model = cls.create(cfg=cfg.flow_model)

        # Load flow model parameters
        actor_path = os.path.join(dir, f"flow_params_{step}.msgpack")
        with open(actor_path, "rb") as f:
            actor_params = flax.serialization.from_bytes(model.target_flow_model.params, f.read())
        model = model.replace(target_flow_model=model.target_flow_model.replace(params=actor_params))

        print(f"Model loaded from {dir}")
        return model