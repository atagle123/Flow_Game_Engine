import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Sequence
import jax

default_init = nn.initializers.xavier_uniform


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

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


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init(), dtype=jnp.float16)(x) # TODO change to 32

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x
    
class ConvBlock(nn.Module):
    out_channels: int
    time_emb_dim: int
    down: bool

    @nn.compact
    def __call__(self, x, t_emb):
        t_proj = nn.Dense(self.out_channels)(t_emb)
        t_proj = t_proj[:, None, None, :]  # reshape for broadcasting

        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = x + t_proj
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        if self.down:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        else:
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method='nearest')
        return x

    
class SimpleUNet(nn.Module):
    def setup(self):

        self.down1 = ConvBlock(32, 64, down=True)
        self.down2 = ConvBlock(64, 64, down=True)
        self.up1 = ConvBlock(32, 64, down=False)
        self.up2 = ConvBlock(3, 32, down=False)

    def __call__(self, x, t_emb):
        x = jnp.transpose(x, (0, 2, 3, 1))  # to (B, W, H, C)

        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        u1 = self.up1(d2, t_emb)
        u2 = self.up2(u1, t_emb)
        out = jnp.transpose(u2, (0, 3, 1, 2))  # back to (B, C, W, H)

        return out