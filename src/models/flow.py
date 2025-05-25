import os
import jax
import jax.numpy as jnp
import flax.serialization
from flax import struct
from flax.training.train_state import TrainState
from typing import Dict, Tuple, Union
import optax
import flax.linen as nn
from src.models.nets import UNet
from src.utils.custom_types import DatasetDict, PRNGKey


def interpolant(
    x0: jnp.ndarray, x1: jnp.ndarray, t: Union[float, jnp.ndarray]
) -> jnp.ndarray:
    return x0 + (x1 - x0) * t


velocity = jax.jacrev(interpolant, argnums=2)

interpolant = jax.vmap(interpolant)
velocity = jax.vmap(velocity)


def count_params(params):
    return sum(jnp.prod(jnp.array(v.shape)) for v in jax.tree_util.tree_leaves(params))


class FlowLearner(struct.PyTreeNode):
    flow_model: TrainState
    target_flow_model: TrainState
    flow_tau: float
    rng: PRNGKey

    @classmethod
    def create(cls, seed: int, cfg):

        rng = jax.random.PRNGKey(seed)
        rng, flow_key = jax.random.split(rng, 2)

        ### Flow model ###

        flow_model_def = UNet(base_channels=cfg.network.base_channels)

        time = jnp.zeros((1, 1))
        x_dummy = jnp.zeros((1, 3, 12, 12))
        actions = jnp.zeros((1, 1))

        flow_params = flow_model_def.init(flow_key, x_dummy, actions, time)["params"]

        print(f"PARAMS: {count_params(flow_params)}")

        if cfg.train.cosine_decay:
            lr = optax.cosine_decay_schedule(
                cfg.train.lr, int(cfg.train.steps), alpha=0.01
            )
        else:
            lr = cfg.train.lr

        flow_model = TrainState.create(
            apply_fn=flow_model_def.apply,
            params=flow_params,
            tx=optax.adam(learning_rate=lr),
        )

        target_flow_model = TrainState.create(
            apply_fn=flow_model_def.apply,
            params=flow_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        return cls(
            flow_model=flow_model,
            target_flow_model=target_flow_model,
            rng=rng,
            flow_tau=cfg.train.ema_update,
        )

    def update_model(
        model, batch: DatasetDict
    ) -> Tuple[struct.PyTreeNode, Dict[str, float]]:
        rng = model.rng
        key, rng = jax.random.split(rng, 2)

        x0 = batch["states"]
        x1 = batch["next_states"]
        actions = batch["actions"]
        actions = jnp.expand_dims(actions, axis=-1)

        t = jax.random.uniform(key, (x1.shape[0], 1))
        key, rng = jax.random.split(rng, 2)

        xt = interpolant(x0, x1, t)
        vt = velocity(x0, x1, t)
        vt = jnp.squeeze(vt, axis=-1)

        def flow_loss_fn(flow_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vt_pred = model.flow_model.apply_fn(
                {"params": flow_model_params}, xt, actions, t
            )

            loss = jnp.mean((vt_pred - vt) ** 2)
            return loss, {"loss": loss}

        grads, info = jax.grad(flow_loss_fn, has_aux=True)(model.flow_model.params)
        flow_model = model.flow_model.apply_gradients(grads=grads)

        model = model.replace(flow_model=flow_model)
        target_flow_params = optax.incremental_update(
            flow_model.params, model.target_flow_model.params, model.flow_tau
        )

        target_flow_model = model.target_flow_model.replace(params=target_flow_params)
        new_model = model.replace(
            flow_model=flow_model, target_flow_model=target_flow_model, rng=rng
        )

        return new_model, info

    @jax.jit
    def update(self, batch: DatasetDict):
        new_model = self
        new_model, info = new_model.update_model(batch)
        return new_model, info

    def sample(self, observation: jnp.ndarray, action: jnp.ndarray, n_steps: int = 100):

        observation = jax.device_put(observation)
        action = jax.device_put(action)
        action = jnp.expand_dims(action, axis=0)

        flow_params = self.target_flow_model.params
        timesteps = jnp.linspace(0, 1, n_steps)

        def flow_step(xt, t_pair):
            t_current, t_next = t_pair
            t = jnp.ones(1) * t_current
            t = jnp.expand_dims(t, axis=0)
            dxt = self.flow_model.apply_fn({"params": flow_params}, xt, action, t)

            xt_new = xt + (t_next - t_current) * dxt
            return xt_new, None

        @jax.jit
        def run_flow(xt_init):
            xt_final, _ = jax.lax.scan(
                flow_step, xt_init, (timesteps[:-1], timesteps[1:])
            )
            return xt_final

        xt_final = run_flow(observation)

        return xt_final

    def save(self, savepath, step):

        actor_path = os.path.join(savepath, f"flow_params_{step}.msgpack")
        with open(actor_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.flow_model.params))

    @classmethod
    def load(cls, cfg, dir: str, step):
        """Load the parameters of the flow model"""
        # Create a new instance of the learner
        model = cls.create(cfg=cfg.flow_model, seed=123)

        # Load flow model parameters
        actor_path = os.path.join(dir, f"flow_params_{step}.msgpack")
        with open(actor_path, "rb") as f:
            actor_params = flax.serialization.from_bytes(
                model.flow_model.params, f.read()
            )
        model = model.replace(
            target_flow_model=model.target_flow_model.replace(params=actor_params)
        )

        print(f"Model loaded from {dir}")
        return model
