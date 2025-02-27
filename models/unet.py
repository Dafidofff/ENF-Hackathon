import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

class UNet(nn.Module):
    """A simple UNet implementation using Flax Linen with time and label conditioning."""
    num_classes: int = 1
    features: int = 32
    layers: int = 4  # Renamed 'depth' to 'layers' for clarity
    time_embed_dim: int = 32 * 4  # Dimension for time embeddings
    num_classes_label: int = 10 # Number of classes for label conditioning (example, CIFAR-10)
    label_embed_dim: int = 32 * 4 # Dimension for label embeddings (can be same as time_embed_dim or different)

    @nn.compact
    def __call__(self, x, t, y, train: bool = True):
        """
        Applies the UNet model with time and label conditioning.

        Args:
            x: Input tensor of shape (batch, height, width, channels).
            t: Time conditioning tensor of shape (batch,). Values should be in [0, 1].
            y: Label conditioning tensor of shape (batch,). Integer class labels.
            train: boolean, whether the model is in training mode or not.

        Returns:
            Output tensor of shape (batch, height, width, num_classes).
        """
        # 1. Time embedding
        time_embed = nn.Dense(self.time_embed_dim)(self.sinusoidal_embedding(t))
        time_embed = nn.relu(time_embed)
        time_embed = nn.Dense(self.time_embed_dim)(time_embed)
        time_embed = nn.relu(time_embed)

        # 2. Label embedding
        label_embed = nn.Embed(num_embeddings=self.num_classes_label, features=self.label_embed_dim)(y)
        label_embed = nn.Dense(self.label_embed_dim)(label_embed)
        label_embed = nn.relu(label_embed)
        label_embed = nn.Dense(self.label_embed_dim)(label_embed)
        label_embed = nn.relu(label_embed)


        skips = []
        for i in range(self.layers):
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)

            # 3. Time conditioning injection
            time_inject = nn.Dense(self.features * (2**i))(time_embed)
            time_inject = jnp.reshape(time_inject, (time_inject.shape[0], 1, 1, time_inject.shape[-1]))

            # 4. Label conditioning injection (similar to time)
            label_inject = nn.Dense(self.features * (2**i))(label_embed)
            label_inject = jnp.reshape(label_inject, (label_inject.shape[0], 1, 1, label_inject.shape[-1]))


            x = x + time_inject + label_inject # Combine time and label injection by addition

            skips.append(x)
            if i < self.layers - 1:
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        for i in range(self.layers - 2, -1, -1):
            x = nn.ConvTranspose(self.features * (2**i), kernel_size=(2, 2), strides=(2, 2))(x)
            x = nn.relu(x)

            # 5. Time conditioning injection (again in upsampling)
            time_inject = nn.Dense(self.features * (2**i))(time_embed)
            time_inject = jnp.reshape(time_inject, (time_inject.shape[0], 1, 1, time_inject.shape[-1]))

            # 6. Label conditioning injection (again in upsampling)
            label_inject = nn.Dense(self.features * (2**i))(label_embed)
            label_inject = jnp.reshape(label_inject, (label_inject.shape[0], 1, 1, label_inject.shape[-1]))

            x = x + time_inject + label_inject # Combine time and label injection by addition


            x = jnp.concatenate([x, skips[i]], axis=-1)
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.Conv(self.features * (2**i), kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)

        x = nn.Conv(self.num_classes, kernel_size=(1, 1))(x)
        return x

    def sinusoidal_embedding(self, t, max_freq=10.0, min_freq=1.0):
        """
        Generates sinusoidal embeddings for time conditioning.

        Args:
            t: Time values (batch,). Values should be in [0, 1].
            max_freq: Maximum frequency for embeddings.
            min_freq: Minimum frequency for embeddings.

        Returns:
            Time embedding tensor (batch, time_embed_dim).
        """
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(min_freq),
                jnp.log(max_freq),
                self.time_embed_dim // 2
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * t[..., None]),
             jnp.cos(angular_speeds * t[..., None])],
            axis=-1
        )
        return embeddings


# Example Usage
def main():
    key = random.PRNGKey(0)
    input_shape = (2, 128, 128, 1)  # Example batch size 2
    inputs = random.normal(key, input_shape)
    time_values = jnp.ones((input_shape[0],)) * 0.5  # Example time value
    label_values = jnp.array([0, 1]) # Example labels for batch size 2 (class indices, e.g., 0 and 1)


    # Example with different number of layers
    model_3_layers = UNet(num_classes=1, layers=3, num_classes_label=10) # Specify num_classes_label
    variables_3_layers = model_3_layers.init(key, inputs, time_values, label_values) # Pass label_values during init
    outputs_3_layers = model_3_layers.apply(variables_3_layers, inputs, time_values, label_values) # Pass label_values during apply
    print("Output shape (3 layers):", outputs_3_layers.shape)

    model_5_layers = UNet(num_classes=1, layers=5, num_classes_label=10) # Specify num_classes_label
    variables_5_layers = model_5_layers.init(key, inputs, time_values, label_values) # Pass label_values during init
    outputs_5_layers = model_5_layers.apply(variables_5_layers, inputs, time_values, label_values) # Pass label_values during apply
    print("Output shape (5 layers):", outputs_5_layers.shape)

    model_default_layers = UNet(num_classes=1, num_classes_label=10) #uses default layers parameter, which is 4, specify num_classes_label
    variables_default_layers = model_default_layers.init(key, inputs, time_values, label_values) # Pass label_values during init
    outputs_default_layers = model_default_layers.apply(variables_default_layers, inputs, time_values, label_values) # Pass label_values during apply
    print("Output shape (default 4 layers):", outputs_default_layers.shape)

if __name__ == "__main__":
    main()