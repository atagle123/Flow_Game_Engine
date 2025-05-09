from typing import Type
import flax.linen as nn
import jax.numpy as jnp

class FlowModel(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]
    action_encoder_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
              # Process condition
        cond = jnp.concatenate([a, cond], axis=-1)
        cond_embedding = self.action_encoder_cls()(cond) # (B,D)
        cond_map = jnp.expand_dims(cond_embedding, axis=-1)  # (B, D, 1)
        cond_map = jnp.expand_dims(cond_map, axis=-1)        # (B, D, 1, 1)
        cond_map = jnp.tile(cond_map, (1, 1, 12, 12))        # (B, D, H, W)

        x = jnp.concatenate([s, cond_map], axis=1)  # (B, C+D, H, W)
       # else:
        #    x = x + cond_map  # requires C == D
        #reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(x, training=training)