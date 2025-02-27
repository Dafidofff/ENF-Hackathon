import jax.linen as nn
import jax.numpy as jnp


class SimpleCNN(nn.Module):
    num_classes: int
    features: list

    @nn.compact
    def __call__(self, x):
        for features in self.features:
            x = nn.Conv(features, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x
