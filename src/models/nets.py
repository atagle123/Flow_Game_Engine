import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Sequence
import jax

default_init = nn.initializers.xavier_uniform


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
    
class ResBlock(nn.Module):
    in_ch: int
    out_ch: int
    emb_dim: int

    @nn.compact
    def __call__(self, x, emb):
        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding='SAME')(x)
        h = nn.silu(h)

        # Add time+cond embedding
        emb_out = nn.Dense(self.out_ch)(emb)
        emb_out = emb_out[:, None, None, :]  # expand for HxW
        h = h + emb_out

        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding='SAME')(nn.silu(h))

        # Skip connection
        skip = x
        if self.in_ch != self.out_ch:
            skip = nn.Conv(self.out_ch, kernel_size=(1, 1))(x)

        return h + skip
    
class UNet(nn.Module):
    in_channels: int = 3
    base_channels: int = 64

    @nn.compact
    def __call__(self, x, cond, t):
        x = jnp.transpose(x, (0, 2, 3, 1))  # to (B, W, H, C)
        time_emb = FourierFeatures(self.base_channels)(t)
        cond_emb = nn.Dense(self.base_channels)(cond)
        emb = time_emb + cond_emb  # shape (B, base_channels)

        # Downsample
        x1 = ResBlock(self.in_channels, self.base_channels, self.base_channels)(x, emb)
        x2 = nn.max_pool(x1, (2, 2))
        x2 = ResBlock(self.base_channels, self.base_channels * 2, self.base_channels)(x2, emb)

        # Bottleneck
        h = nn.max_pool(x2, (2, 2))
        h = ResBlock(self.base_channels * 2, self.base_channels * 2, self.base_channels)(h, emb)

        # Upsample
        h = jax.image.resize(h, shape=(h.shape[0], x2.shape[1], x2.shape[2], h.shape[3]), method='nearest')
        h = jnp.concatenate([h, x2], axis=-1)
        h = ResBlock(self.base_channels * 4, self.base_channels, self.base_channels)(h, emb)

        h = jax.image.resize(h, shape=(h.shape[0], x1.shape[1], x1.shape[2], h.shape[3]), method='nearest')
        h = jnp.concatenate([h, x1], axis=-1)
        h = ResBlock(self.base_channels * 2, self.base_channels, self.base_channels)(h, emb)
        out = nn.Conv(self.in_channels, kernel_size=(1, 1))(h)

        return jnp.transpose(out, (0, 3, 1, 2))  # back to (B, C, W, H)