# Adapted from https://github.com/kvfrans/jax-flow/tree/main, by Kevin Frans
import flax.linen as nn
import jax.numpy as jnp


from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6))(inputs)
        x = nn.gelu(x)
        output = nn.Dense(
                features=actual_out_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6))(x)
        return output


def modulate(x, shift, scale):
    """
    Apply modulation to input features using shift and scale parameters.
    
    This function implements feature-wise linear modulation (FiLM),
    which conditions neural networks by applying an affine transformation
    to intermediate features.
    
    Args:
        x: Input tensor to be modulated
        shift: Additive shift parameter (bias)
        scale: Multiplicative scale parameter
        
    Returns:
        Modulated features: x * (1 + scale) + shift, with proper broadcasting
    """
    return x * (1 + scale[:, None]) + shift[:, None]


class TransformerBlock(nn.Module):
    """
    A Transformer block implementation with self-attention and feed-forward layers.
    
    This block follows the architecture from the "Attention Is All You Need" paper,
    using LayerNorm, Multi-Head Attention, and MLP components with residual connections.
    
    Attributes:
        hidden_size: Dimension of the input and output features
        num_heads: Number of attention heads
        mlp_ratio: Multiplier for the hidden dimension in the MLP block (default: 4.0)
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        # Layer Normalization before self-attention
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        
        # Self-attention block with residual connection
        # Query, key, and value are all the same input (x_norm)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads
        )(x_norm, x_norm)
        x = x + attn_x  # Add residual connection
        
        # Layer Normalization before MLP
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        
        # MLP block with expanded hidden dimension
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_norm2)
        x = x + mlp_x  # Add residual connection
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_channels, kernel_init=nn.initializers.constant(0))(x)
        return x


class PosEmb(nn.Module):
    """ RFF positional embedding. """
    embedding_dim: int
    freq: float

    @nn.compact
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        emb = nn.Dense(self.embedding_dim // 2, kernel_init=nn.initializers.normal(self.freq), use_bias=False)(
            jnp.pi * (coords + 1))  # scale to [0, 2pi]
        return nn.Dense(self.embedding_dim)(jnp.sin(jnp.concatenate([coords, emb, emb + jnp.pi / 2.0], axis=-1)))


class TransformerClassifier(nn.Module):
    """
    Transformer model with a final layer for classification.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    num_classes: int

    @nn.compact
    def __call__(self, p_0, c_0, g_0):
        # Embed patched and poses.
        pos_embed = PosEmb(self.hidden_size, freq=1.0)(p_0)
        c = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(c_0)
        c = c + pos_embed

        # Run DiT blocks on input and conditioning.
        for _ in range(self.depth):
            c = TransformerBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(c)

        # Final layer.
        return FinalLayer(self.num_classes, self.hidden_size)(jnp.mean(c, axis=1))


class TransformerForecaster(nn.Module):
    """
    Transformer model specific for forecasting. This model predicts the next latent point cloud in a sequence.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    learn_pose: bool = True

    @nn.compact
    def __call__(self, p_0, c_0, g_0):
        
        in_channels = c_0.shape[-1]
        if self.learn_pose:
            out_channels = in_channels + 2
        else:
            out_channels = in_channels

        # Embed patched and poses.
        pos_embed = PosEmb(self.hidden_size, freq=1.0)(p_0)
        c = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(c_0)
        c = c + pos_embed

        # Run DiT blocks on input and conditioning.
        for _ in range(self.depth):
            c = TransformerBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(c)

        # Final layer.
        c = FinalLayer(out_channels, self.hidden_size)(c)

        # Split into p, c if learning pose
        if self.learn_pose:
            c, p = c[..., :-2], c[..., -2:]
            return p_0 + p, c_0 + c, g_0
        else:
            return p_0, c_0 + c, g_0