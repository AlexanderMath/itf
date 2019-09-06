import sys
import os 
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.math import l2_normalize as normalize
import numpy as np
from tensorflow.keras.layers import ReLU, Conv2D
from invtf.override import print_summary
from invtf.coupling_strategy import *

# True is column false is row
vectorize = lambda x, t: tf.reshape(x, (tf.squeeze(x).shape[0], 1)) if t else tf.reshape(x, (1, tf.squeeze(x).shape[0]))

def _l2normalize(v, eps=1e-12):
	return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

class SpectralNormalizationConstraint(keras.constraints.Constraint):
	def __init__(self, coeff, input_shape, n_power_iter=5, strides=None, padding=None, **kwargs):
		"""
			Arguments:
				coeff:			Contraction coefficient of the conv (iResNet default is 0.9)
				input_shape:	Input_shape of the layer to be normalized
				n_power_iter:	Number of power iterations to estimate largest eigenvalue (iResNet default is 0.9)
		"""
		# TODO this implementation does not use the trick for minimizing iterations
		# by only doing one step for each gradient step.
		# Not sure how many power iterations is needed? 5 is default in iResNet.
		super(SpectralNormalizationConstraint, self).__init__(**kwargs)
		self.coeff = coeff
		self.n_power_iter = n_power_iter if n_power_iter >= 0 else 5
		self.input_shape = input_shape
		self.strides = strides
		self.padding = padding

	def _power_iteration_conv(self, w):
		params	= {'strides': self.strides, 'padding': self.padding}

		input_shape		= [1] + [int(i) for i in self.input_shape[1:]]

		# initialize u and v to norm 1 random vectors and calc output shape
		v				= _l2normalize(tf.random.normal((tf.reduce_prod(input_shape),)))

		tmp				= tf.nn.conv2d(tf.reshape(v, input_shape), w, **params)
		output_shape	= tmp.shape

		u				= _l2normalize(tf.random.normal((tf.reduce_prod(output_shape),)))

		# Helper reshape functions
		inp		= lambda v: tf.reshape(v, input_shape)	# Reshape to input_shape
		out		= lambda u: tf.reshape(u, output_shape)	# Reshape to output_shape
		flat	= lambda x: tf.reshape(x, [-1])			# Flatten
		
		# Power iteration
		for _ in range(self.n_power_iter):
			v_s	= tf.nn.conv2d_transpose(out(u), w, output_shape=input_shape, **params)
			v	= _l2normalize(flat(v_s))

			u_s = tf.nn.conv2d(inp(v), w, **params)
			u	= _l2normalize(flat(u_s))

		# Normalization v @ W @ u^T
		vW		= tf.nn.conv2d(inp(v), w, **params)
		sigma	= tf.tensordot(flat(vW), u, 1)

		return tf.maximum(1., sigma / self.coeff)

	def _power_iteration_dense(self, w):
		def power_iteration(W, u):
			#Accroding the paper, we only need to do power iteration one time for each iteration.
			#However, we cannot store weights in Contraints, so we cannot use that trick.
			_u = u
			_v = _l2normalize( vectorize(_u, False) @ tf.transpose(W))
			_u = _l2normalize( _v @ W )
			return _u, _v

		u = tf.random.normal((w.shape[-1],))
		for i in range(self.n_power_iter): 
			u, v = power_iteration(w, u)

		sigma = vectorize(v, False) @ w @ vectorize(u, True)
		return tf.maximum(1., sigma / self.coeff)
 
	def _call_include_sigma(self, w): # This function is mainly used for tests of constraints ( to be able to extract sigma )
		factor1 = None
		if self.strides: # Conv
			factor = self._power_iteration_conv(w)
		else:
			factor = self._power_iteration_dense(w)
			
		res = w / (factor + 1e-5) # Stability
		return res, factor

	def __call__(self, w):
		scaled_weights, sigma = self._call_include_sigma(w)
		return scaled_weights


class ResidualBlock(keras.layers.Layer):
	"""
		TODO: fill in some more proper text with references, code examples, etc.
		Residual blocks are fra [ResNet] and [iResNet] showed how 
		there exists an inverse of ResNets if each residual block
			x_{t+1} = x_t + g_t(x_t)
		has functions g_t with the Lipschitz norm ||g_t||_L < 1.
		The gs are typically parametrized by neural networks and
		employ spectral normalization [..] to ensure the Lipschitz
		constraint.
	"""
	unique_id = 1

	def __init__(self, **kwargs):
		super(ResidualBlock, self).__init__(name="res_block_%i"%ResidualBlock.unique_id, **kwargs)
		ResidualBlock.unique_id += 1
		self.layers				= []

	def add(self, layer): self.layers.append(layer)

	def build(self, input_shape, *args, **kwargs):
		out_dim = input_shape
		for layer in self.layers:
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)

		super(ResidualBlock, self).build(input_shape, *args, **kwargs)
		self.built = True

	def _call(self, X):
		out = X
		for layer in self.layers:
			out = layer(out)
		return out

	def call(self, X):
		gx = self._call(X)
		return X + gx

	def call_inv(self, Z):
		# TODO: no stopping early here.  Should there be some threshold on how
		# much the solution changes?
		maxIter = 100 # Arbitrary max iteration for inverse (choosen equal to default in iResNet # TODO Make dynamic
		X		= Z
		for i in range(maxIter):
			gx = self._call(X)
			X = Z - gx
		return X

	@tf.function
	def log_det(self):	
		"""
		TODO:	We produce `N` random vectors and do the power series for all of them
				and then we take the mean of these. This is not really described in the 
				article [.. iResNet TODO ..] but it is done in their code `matrix_utils.py:94`.
		"""
		print("Log det!")
		bs		= 1		# Fixed batch size for now TODO: make dynamic
		N		= 10	# Number of trace samples  TODO: make dynamic
		K		= 5		# Number of Series terms	TODO: make dynamic

		print(self.input)
		X		= self.input
		# X = tf.random.normal((1, 32, 32, 3))

		# Shape (b, N, h, w, c)
		dim		= [bs, N] + list(X.shape[1:])
		V		= tf.random.normal(dim)

		# Will end up being shape (bs, N, 1)
		trLn	= 0.
		for j in range(K): 
			if j == 0:
				W = tf.identity(V)
			# Due to chain rule, taking gradient of W is the same as wT @ J
			W_list	= []
			for i in range(N):
				with tf.GradientTape() as tape:
					tape.watch(X)
					Z_ = self.call(X)
				grads = tape.gradient(Z_, X, output_gradients=W[:,i])
				W_list.append(grads)

			# W		= [tf.gradients(Z, X, grad_ys=W[:,i]) for i in range(N)]
			W		= tf.stack(W_list, axis=1)
			print("W:", W.shape)
			wT_J	= tf.reshape(W, (-1, N, 1, tf.reduce_prod(X.shape[1:])))
			print("WT_J:", wT_J.shape)
			vFlat	= tf.reshape(V,	(-1, N, tf.reduce_prod(X.shape[1:]), 1))
			print("vFlat:", vFlat.shape)

			# Shape (b, N, 1)
			product	= wT_J @ vFlat / float(K) 
			print("Product:", product.shape)
			tfLn = trLn + product * (1. if (K+1) % 2 == 0 else -1.)

		print("tfLn final:", tfLn.shape)
		return tf.squeeze(tf.reduce_mean(tfLn, axis=1))

