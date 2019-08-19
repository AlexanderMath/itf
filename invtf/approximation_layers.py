import os 
import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.layers import ReLU, Conv2D
from invtf.override import print_summary
from invtf.coupling_strategy import *


class SpectralNormalizationConstraint(keras.constraints.Constraint):
	def __call__(self, w, n_power_iter=5):
		# TODO do spectral normalization.
		# Not sure how many power iterations is needed? 5 is default in iResNet.
		return w

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

	def __init__(self):
		super(ResidualBlock, self).__init__(name="res_block_%i"%ResidualBlock.unique_id)
		ResidualBlock.unique_id += 1

		self.layers				= []

	def add(self, layer): self.layers.append(layer)

	def build(self, input_shape):
		out_dim = input_shape
		for layer in self.layers:
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)

		super(ResidualBlock, self).build(*args, **kwargs)
		self.built = True

	def _call(self, X):
		out = X
		for layer in self.layers:
			out = layer.call(out)
		return out

	def call(self, X):
		gx = self._call(X)
		return X + gx

	def call_inv(self, Z):
		# TODO: no stopping early here.  Should there be some threshold on how
		# much the solution changes?
		maxIter = 100 # Arbitrary max iteration for inverse (choosen equal to default in iResNet # TODO Make dynamic
		X = Z
		for i in arange(maxIter):
			gx = self._call(X)
			X = Z - gx
		return X

	def log_det(self, X, Z):	
		"""
		TODO:	We produce `N` random vectors and do the power series for all of them
				and then we take the mean of these. This is not really described in the 
				article [.. TODO ..] but it is done in their code `matrix_utils.py:94`.
		"""
		N		= 10 # Number of trace samples  TODO: make dynamic
		K		= 5  # Number of Series terms	TODO: make dynamic

		# Shape (b, N, h, w, c)
		dim		= [self.input_shape[0], N] + self.input_shape[1:]
		V		= tf.random.normal(dim)

		# Will end up being shape (b, N, 1)
		trLn	= 0.
		for j in range(1, k+1): 
			if j == 1:
				W = v.clone()
			# Due to chain rule, taking gradient of W is the same as wT @ J
			W		= [tf.gradients(Z, X, grad_ys=W[:,i]) for i in range(N)]
			W		= tf.stack(W, axis=1)
			wT_J	= grads.reshape(self.input_shape[0], N, 1, -1)
			vFlat	= v.reshape(self.input_shape[0], N, -1, 1)

			# Shape (b, N, 1)
			product	= wT_J @ vFlat / float(k) 
			tfLn = trLn + product * (1. if (k+1) % 2 == 0 else -1.)

		return tfLn.mean(axis=1).squeeze()

