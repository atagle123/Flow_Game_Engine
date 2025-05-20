from typing import Type
import flax.linen as nn
import jax.numpy as jnp

class FlowModel(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        cond = jnp.concatenate([a, cond], axis=-1)

        return self.reverse_encoder_cls()(s, cond)