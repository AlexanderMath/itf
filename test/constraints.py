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

class SpectralNormalizationTest(unittest.TestCase):

	X = tf.constant(keras.datasets.cifar10.load_data()[0][0][:100].astype('f')) # a single cifar image batch.

	def test_spectral_norm_dense(self):
		snc = SpectralNormalizationConstraint(1., self.X.shape, n_power_iter=10)
		w = tf.random.normal((100, 100))
		new_w, lambda_hat = snc._call_include_sigma(w)
		
		lambda_real = tf.reduce_max(tf.linalg.eigvalsh(w))  # Real eigenvalue

		new_w, lambda_hat = snc._call_include_sigma(new_w)  # Estimated eigenvalue
		self.assertTrue(np.allclose(1.0, lambda_hat, atol=1e-1, rtol=1e-1))

		_, lambda_hat = snc._call_include_sigma(new_w)	  # Afterwards, w should have norm roughly 1
		self.assertTrue(np.allclose(1.0, lambda_hat, atol=1e-1, rtol=1e-1), lambda_hat.numpy())

	def test_spectral_norm_conv(self):
		snc = SpectralNormalizationConstraint(1., self.X.shape, n_power_iter=20, strides=[1,1], padding='SAME')
		w = tf.random.normal((100, 100, 3, 3))
		new_w, lambda_hat = snc._call_include_sigma(w)	  # Normalize w
		_, lambda_hat = snc._call_include_sigma(new_w)	  # Afterwards, w should have norm roughly 1

		self.assertTrue(np.allclose(1.0, lambda_hat, atol=1e-1, rtol=1e-1), lambda_hat.numpy())


