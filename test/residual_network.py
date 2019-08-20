import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.math import l2_normalize as normalize

import numpy as np
import matplotlib.pyplot as plt

# Local lib
import sys
sys.path.append("../")
from invtf.approximation_layers import ResidualBlock, SpectralNormalizationConstraint

class ResidualNetworkTest(unittest.TestCase):
	X = tf.constant(keras.datasets.cifar10.load_data()[0][0][:100].astype('f')) # a single cifar image batch.

	def get_norm(self, w):
		input_shape		= [1] + [int(i) for i in self.X.shape[1:]]

		# initialize u and v to norm 1 random vectors
		v				= normalize(tf.random.normal((tf.reduce_prod(input_shape),)))

		tmp				= tf.nn.conv2d(tf.reshape(v, input_shape), w, strides=[1,1], padding='SAME')
		output_shape	= tmp.shape

		u				= normalize(tf.random.normal((tf.reduce_prod(output_shape),)))

		inp		= lambda v: tf.reshape(v, input_shape)
		out		= lambda u: tf.reshape(u, output_shape)
		flat	= lambda x: tf.reshape(x, [-1])
		params	= {'strides': [1,1], 'padding': 'SAME'}
		
		# Power iteration
		for _ in range(5):
			v_s	= tf.nn.conv2d_transpose(out(u), w, output_shape=input_shape, **params)
			v	= normalize(flat(v))

			u_s = tf.nn.conv2d(inp(v), w, **params)
			u	= normalize(flat(u))

		# Normalization
		vW		= tf.nn.conv2d(inp(v), w, **params)
		sigma	= tf.tensordot(flat(vW), u, 1)
		factor	= sigma / 0.9 # tf.maximum(1., sigma / 0.9)
		return sigma, factor


	def test_inverse(self):
		r = ResidualBlock()

		sn = SpectralNormalizationConstraint(0.9, self.X.shape, n_power_iter=5, strides=[1,1], padding='SAME')
		r.add(Conv2D(3, kernel_size=(3,3), activation='elu',      padding='SAME', strides=[1,1], kernel_constraint=sn)) 
		sn = SpectralNormalizationConstraint(0.9, self.X.shape, n_power_iter=5, strides=[1,1], padding='SAME')
		r.add(Conv2D(1, kernel_size=(3,3), activation='elu',      padding='SAME', strides=[1,1], kernel_constraint=sn)) 
		sn = SpectralNormalizationConstraint(0.9, [int(i) for i in self.X.shape[:-1]] + [1], n_power_iter=5, strides=[1,1], padding='SAME')
		r.add(Conv2D(3, kernel_size=(3,3), activation=None,      padding='SAME', strides=[1,1], kernel_constraint=sn)) 

		s = Sequential()
		s.add(r)
		optimizer = Adam()
		loss = MeanSquaredError()
		s.compile(optimizer=optimizer, loss=loss)
		Z = s.predict(self.X)

		print('\n\n\n - - -  \n\n\n')
		print(self.get_norm(r.layers[0].kernel))
		print('\n\n\n - - -  \n\n\n')

		s.fit(self.X, self.X, epochs=10)

		Xr = r.call_inv(Z)
		A = np.allclose(self.X, Xr, atol=1e-3, rtol=1e-3)

		print('\n\n\n - - -  \n\n\n')
		print(self.get_norm(r.layers[0].kernel))
		print('\n\n\n - - -  \n\n\n')

		# Plotting stuff
		if not A:
			rows=3
			fig, ax = plt.subplots(rows + 1, 4)
			ax[0, 0].set_title('X')
			ax[0, 1].set_title('Rec')
			ax[0, 2].set_title('Diff')
			ax[0, 3].set_title('Z')
			for i in range(rows):
				ax[i, 0].imshow(tf.reshape(self.X[i], (32, 32, 3)) / 255.)
				ax[i, 1].imshow(tf.reshape(Xr[i], (32, 32, 3)) / 255.)
				ax[i, 2].imshow(tf.reshape(Xr[i] - self.X[i], (32, 32, 3)) / 255.)
				ax[i, 3].imshow(tf.reshape(Z[i], (32, 32, 3)) / 255.)

			entry = tf.reduce_max(tf.abs(self.X - Xr))
			truth = tf.equal(tf.abs(self.X - Xr), entry)
			i = tf.argmax(tf.cast([tf.reduce_any(truth[i]) for i in range(len(self.X))], tf.float32))
			ax[rows, 0].imshow(tf.reshape(self.X[i], (32, 32, 3)) / 255.)
			ax[rows, 1].imshow(tf.reshape(Xr[i], (32, 32, 3)) / 255.)
			ax[rows, 2].imshow(tf.reshape(Xr[i] - self.X[i], (32, 32, 3)) / 255.)
			ax[rows, 3].imshow(tf.reshape(Z[i], (32, 32, 3)) / 255.)
			plt.show()

		assert A, tf.reduce_max(tf.abs(self.X - Xr))

