#! /usr/bin/env python

from typing import Any, Callable

import chex
import flax
import flax.struct
import jax.numpy as jnp
import flax.linen as nn
import math


def shift_right(x, axis=1):
	"""Shift the input to the right by padding on axis 1."""
	pad_widths = [(0, 0)] * len(x.shape)
	pad_widths[axis] = (1, 0)
	padded = jnp.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
	return padded[:, :-1]


@chex.dataclass
class TransformerConfig:
	
	vocab_size: int
	out_vocab_size: int
	num_heads: int = 8
	num_layers: int = 6
	num_embeds: int = 768
	dropout_rate: float = 0.1
	attention_dropout_rate: float = 0.1
	use_bias: bool = True
	dtype: Any = jnp.float32
	act_fn: Callable = nn.gelu
	qkv_dim: int = 512
	mlp_dim: int = 2048
	max_len: int = 2048
	mlp_layers: int = 2
	kernel_init: Callable = nn.initializers.xavier_uniform()
	bias_init: Callable = nn.initializers.normal(stddev=1e-6)
	embed_cls: Callable = flax.linen.Embed
	embed_kw: dict[str, Any] = flax.struct.field(
		pytree_node=False,
		default_factory=lambda: dict(
			num_embeddings=1, features=512, embedding_init=flax.linen.initializers.normal(stddev=1.0)
		),
	)


class PositionalEncoding(nn.Module):
	mlp_dim : int         # Hidden dimensionality of the input.
	max_len : int		  # Maximum length of a sequence to expect.
	
	def setup(self):
		# Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
		pe = jnp.zeros((self.max_len, self.mlp_dim))
		position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:,None]
		div_term = jnp.exp(jnp.arange(0, self.mlp_dim, 2) * (-math.log(10000.0) / self.mlp_dim))
		pe[:, 0::2] = jnp.sin(position * div_term)
		pe[:, 1::2] = jnp.cos(position * div_term)
		self.pe = pe[None]
	
	def __call__(self, x):
		x = x + self.pe[:, :x.shape[1]]
		return x


class MLPBlock(nn.Module):
	
	config: TransformerConfig
	out_dim: int | None = None
	
	def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
		
		out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
		dtype = self.config.dtype
		kernel_init = self.config.kernel_init
		bias_init = self.config.bias_init
		
		for _ in range(self.config.mlp_layers - 1):
			x = self.config.act_fn(nn.Dense(self.config.mlp_dim, dtype=dtype, kernel_init=kernel_init, bias_init=bias_init)(x))
			x = nn.Dropout(self.config.dropout_rate)(x, deterministic=not train)
		
		x = nn.Dense(out_dim, dtype=dtype, kernel_init=kernel_init, bias_init=bias_init)(x)
		x = nn.Dropout(self.config.dropout_rate)(x, deterministic=not train)
		return x
	

class TransformerBlock(nn.Module):
	config: TransformerConfig
	
	def __call__(self, x: jnp.ndarray, train: bool = False, mask: chex.Array | None = None) -> jnp.ndarray:
	
		out = nn.LayerNorm(dtype=self.config.dtype, epsilon=1e-5, use_bias=self.config.use_bias, bias_init=self.config.bias_init)(x)
		out = nn.MultiHeadDotProductAttention(
			num_heads=self.config.num_heads,
			dtype=self.config.dtype,
			qkv_features=self.config.qkv_dim,
			kernel_init=self.config.kernel_init,
			bias_init=self.config.bias_init,
			use_bias=self.config.use_bias,
			broadcast_dropout=False,
			dropout_rate=self.config.attention_dropout_rate,
			deterministic=not train,
		)(out, mask=mask)
		out = flax.linen.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
		out += x
		
		residual = out
		out = nn.LayerNorm(dtype=self.config.dtype, epsilon=1e-5, use_bias=self.config.use_bias, bias_init=self.config.bias_init)(out)
		out = MLPBlock(config=self.config)(out, train=train)
		
		return out + residual
	

class EncoderDecoderBlock(nn.Module):
	config: TransformerConfig
	
	def __call__(self, x: jnp.ndarray, encoded: jnp.ndarray, train: bool = False,
				 attn_1_mask: chex.Array | None = None,
				 attn_2_mask: chex.Array | None = None) -> jnp.ndarray:
		out = nn.LayerNorm(dtype=self.config.dtype, epsilon=1e-5, use_bias=self.config.use_bias, bias_init=self.config.bias_init)(x)
		out = nn.MultiHeadDotProductAttention(
			num_heads=self.config.num_heads,
			dtype=self.config.dtype,
			qkv_features=self.config.qkv_dim,
			kernel_init=self.config.kernel_init,
			bias_init=self.config.bias_init,
			use_bias=self.config.use_bias,
			broadcast_dropout=False,
			dropout_rate=self.config.attention_dropout_rate,
			deterministic=not train,
		)(out, mask=attn_1_mask)
		out = flax.linen.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
		out += x
		
		residual = out
		out = nn.LayerNorm(dtype=self.config.dtype, epsilon=1e-5, use_bias=self.config.use_bias, bias_init=self.config.bias_init)(out)
		out = nn.MultiHeadDotProductAttention(
			num_heads=self.config.num_heads,
			dtype=self.config.dtype,
			qkv_features=self.config.qkv_dim,
			kernel_init=self.config.kernel_init,
			bias_init=self.config.bias_init,
			use_bias=self.config.use_bias,
			broadcast_dropout=False,
			dropout_rate=self.config.attention_dropout_rate,
			deterministic=not train,
		)(out, encoded, mask=attn_2_mask)
		out = flax.linen.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
		out += residual
		
		residual = out
		out = nn.LayerNorm(dtype=self.config.dtype, epsilon=1e-5, use_bias=self.config.use_bias, bias_init=self.config.bias_init)(out)
		out = MLPBlock(config=self.config)(out, train=train)
		
		return out + residual
	

class Encoder(nn.Module):

	config: TransformerConfig
	
	# noinspection PyAttributeOutsideInit
	def setup(self) -> None:
		
		self.embed = self.config.embed_cls(**self.config.embed_kw)
		self.positional_encoding = PositionalEncoding(self.config.mlp_dim, self.config.max_len)
		self.transformer_block = TransformerBlock(self.config)
		
	
	def __call__(self, x: jnp.ndarray, train: bool = False, add_positional: bool = True, encoder_mask: chex.Array | None = None) -> jnp.ndarray:
		
		out = x.astype(jnp.int32)
		out = self.embed(out)
		if add_positional:
			out = self.positional_encoding(out)
		out = nn.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
		out = out.astype(self.config.dtype)
		
		for _ in range(self.config.num_layers):
			out = self.transformer_block(out, train, mask=encoder_mask)
		
		out = flax.linen.LayerNorm(dtype=self.config.dtype)(out)
		return out


class Decoder(nn.Module):
	config: TransformerConfig
	
	# noinspection PyAttributeOutsideInit
	def setup(self) -> None:
		
		self.embed = self.config.embed_cls(**self.config.embed_kw)
		self.positional_encoding = PositionalEncoding(self.config.mlp_dim, self.config.max_len)
		self.transformer_block = TransformerBlock(self.config)
	
	def __call__(self, x: jnp.ndarray, train: bool = False, add_positional: bool = True, decode: bool = False,
				 encoder_mask: chex.Array | None = None) -> jnp.ndarray:
		
		out = x.astype(jnp.int32)
		
		if decode:
			out = shift_right(out)
		
		out = self.embed(out)
		if add_positional:
			out = self.positional_encoding(out)
		out = nn.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
		out = out.astype(self.config.dtype)
		
		for _ in range(self.config.num_layers):
			out = self.transformer_block(out, train, mask=encoder_mask)
		
		out = flax.linen.LayerNorm(dtype=self.config.dtype)(out)
		kernel_init = self.config.kernel_init
		bias_init = self.config.bias_init
		out = flax.linen.Dense(self.config.out_vocab_size, dtype=self.config.dtype, kernel_init=kernel_init, bias_init=bias_init)(out)
		return out
	
	

