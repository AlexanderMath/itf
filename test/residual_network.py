import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import unittest

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg

# Local lib
import sys
sys.path.append("../")
from invtf.approximation_layers import ResidualBlock, SpectralNormalizationConstraint

class ResidualBlockTest(unittest.TestCase):
	X = tf.constant(keras.datasets.cifar10.load_data()[0][0][:20].astype('f')) # a single cifar image batch.
	dense_size = 100

	def _assert_inverse(self, residual_block, X):
		sequential = Sequential()
		sequential.add(residual_block)

		optimizer = Adam()
		loss = MeanSquaredError()
		sequential.compile(optimizer=optimizer, loss=loss) 
	 	 
		sequential.fit(X, X)

		Z = sequential.predict(X)
		X_ = residual_block.call_inv(Z) 

		self.assertTrue(np.allclose(X, X_, atol=1e-3, rtol=1e-3), np.max(np.abs(X - X_)))
		
	def test_dense_inverse(self):
		# Flatten input
		X = tf.reshape(ResidualBlockTest.X, (-1, tf.reduce_prod(ResidualBlockTest.X.shape[1:])))[:,:self.dense_size]
		#####
		rb  = ResidualBlock()

		snc = SpectralNormalizationConstraint(0.9, X.shape)
		dense = Dense(self.dense_size, input_shape=X.shape, use_bias=False, kernel_constraint=snc)
		rb.add(dense)
		#####

		self._assert_inverse(rb, X)

	def test_conv_inverse(self):
		#####
		rb  = ResidualBlock()

		snc = SpectralNormalizationConstraint(0.9, ResidualBlockTest.X.shape, strides=[1,1], padding='SAME')
		conv = Conv2D(3, 
				kernel_size=3, 
				input_shape=ResidualBlockTest.X.shape, 
				use_bias=False, 
				kernel_constraint=snc, 
				strides=[1,1], 
				padding='SAME')

		rb.add(conv)
		#####

		self._assert_inverse(rb, ResidualBlockTest.X)

	def test_conv_inverse_deeper(self):
		#####
		rb  = ResidualBlock()
		params = {'strides': [1,1], 'padding': 'SAME'}

		def add_conv(kernels, input_shape):
			snc = SpectralNormalizationConstraint(0.9, input_shape, **params)
			conv = Conv2D(3, kernel_size=kernels, input_shape=input_shape, kernel_constraint=snc, **params)
			rb.add(conv)
			return conv.compute_output_shape(input_shape)

		out_shape = ResidualBlockTest.X.shape
		for i in [3, 1, 3]: 
			out_shape = add_conv(i, out_shape)
		#####

		self._assert_inverse(rb, ResidualBlockTest.X)

