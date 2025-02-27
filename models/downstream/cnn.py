import flax.linen as nn
import jax.numpy as jnp


class ResidualBlock(nn.Module):
    """Residual block for CNN architecture.
    
    Implements a residual connection where the input is added to the output
    of a series of convolutional operations, allowing for better gradient flow
    during training.
    
    Attributes:
        features: Number of output filters in the convolution layers.
    """
    features: int

    @nn.compact
    def __call__(self, x):
        # Store input for residual connection
        residual = x
        
        # First convolution layer followed by ReLU activation
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        
        # Second convolution layer (no activation yet)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(x)

        # Adjust residual dimensions if needed to match output dimensions
        if residual.shape[-1] != x.shape[-1]:
            residual = nn.Conv(self.features, kernel_size=(1, 1), padding="SAME")(residual)

        # Add residual connection and apply ReLU
        x = nn.relu(x + residual)
        return x


class ResidualCNN(nn.Module):
    """Residual CNN model for image classification.
    
    A convolutional neural network with residual connections for better
    gradient flow. Features a series of residual blocks followed by
    pooling operations, and ends with global average pooling and a
    dense classification layer.
    
    Attributes:
        num_classes: Number of output classes for classification.
        features: List of feature dimensions for each residual block.
    """
    num_classes: int
    features: list

    @nn.compact
    def __call__(self, x):
        # Process through each residual block followed by max pooling
        for features in self.features:
            x = ResidualBlock(features=features)(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Final classification layer
        x = nn.Dense(self.num_classes)(x)
        return x