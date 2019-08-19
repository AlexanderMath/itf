import unittest
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt

# Local lib
import sys
sys.path.append("../")
from invtf.approximation_layers import ResidualBlock, SpectralNormalizationConstraint

class ResidualNetworkTest(unittest.TestCase):
	X = tf.constant(keras.datasets.cifar10.load_data()[0][0][:10].astype('f')) # a single cifar image batch.

	def assertInverse(self, g, X): 
		"""
			Input: 		
				g:		Model which has call(X) return a tensor that depends on X. 
				X:		Test data. 
	
			Computes the reconstruction of X and compares with X. 

		""" 
		rec = g.call_inv(X)
		A = np.allclose(rec, X, atol=1, rtol=0.1) # assumes data is in bytes. 

		# find entry with largest difference and print their relative values. 
		diff = np.abs(A - rec)
		entry = np.argmax(diff)

		if A is False: 
			fig, ax = plt.subplots(3, 1)
			ax[0].imshow(rec.reshape(32,32,3)/255)
			ax[1].imshow(X.reshape(32,32,3)/255)
			ax[2].imshow((rec-X).reshape(32,32,3)/255)
			plt.show()

		self.assertTrue(A)
		
	def test_inverse(self):
		r = ResidualBlock()
		#																	( b,     h,      w,     c )
		r.add(Conv2D(3, kernel_size=(3,3), activation="relu",      padding='same',
						kernel_constraint=SpectralNormalizationConstraint())) 

		y = r.call(self.X)
		self.assertInverse(r, y)


