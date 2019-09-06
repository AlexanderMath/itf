
import unittest
import sys
sys.path.append("../")
import invtf
import invtf.layers
import invtf.approximation_layers
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D
import numpy as np
import warnings


"""
	Some of the computations are happening on the CPU and are, as a result, very slow. 
"""

class TestJacobian(unittest.TestCase): 

	X = keras.datasets.cifar10.load_data()[0][0][:1].astype(np.float32) # a single cifar image. 

	def assertJacobian(self, g, X): 
		"""
			Input: 		
				g:		Model which has call(X) return a tensor that depends on X. 
				X:		Test data. 
	
			Computes the jacobian of g(X) wrt X using tf.GradientTape and compares 
			the log determinant with that computed using log_det. 

			The stability of determinant computations are not that good, so the 
			np.allclose has a quite high absolute tolerance. 

		"""
		X = tf.constant(X)
		print("Input before", g.layers[0].input)
		with tf.GradientTape(persistent=True) as t: 
			t.watch(X)
			z = g.call(X)

		print("Input after", g.layers[0].input)
		assert False

		print("Jacobian")
		J = t.jacobian(z, X, experimental_use_pfor=False)
		print(J.shape)

		print("Reshape")
		J = tf.reshape(J, (32*32*3, 32*32*3))

		print("Start Log 1")
		lgdet1 = tf.math.log(tf.linalg.det(J)).numpy()

		# TODO: Probably need a better structure for Jacobian determinants.
		print("Start Log 2")
		lgdet2 = g.log_det()

		print("Look at this: ", lgdet1, lgdet2)
		
		# If the following equation is element-wise True, then allclose returns True.
		# 		absolute(a - b) <= (atol + rtol * absolute(b))
		# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
		A = np.allclose( lgdet1, lgdet2 , atol=10**(-3), rtol=0.15)  # deciding these values are very difficult for all tests.. 

		self.assertTrue(A)
		#print("\t", lgdet1, lgdet2, "\t", end="")
		
	def execute_with_session(op):
		pass


	def test_actnorm_init(self): 
		X = TestJacobian.X
		g = invtf.Generator() 
		g.add(invtf.layers.ActNorm(input_shape=X.shape[1:])) 
		g.compile()
		g.init(X[:100])
		self.assertJacobian(g, X)

	def test_actnorm_fit(self): 
		X = TestJacobian.X
		g = invtf.Generator() 
		g.add(invtf.layers.ActNorm(input_shape=X.shape[1:])) 
		g.compile()
		g.init(X[:100])
		g.fit(X[:1], verbose=False)
		self.assertJacobian(g, X)


	def test_invconv_init(self): 
		X = TestJacobian.X
		g = invtf.Generator() 
		g.add(invtf.layers.Inv1x1Conv(input_shape=X.shape[1:])) 
		g.compile()
		g.init(X[:100])

		self.assertJacobian(g, X)

	def test_invconv_fit(self): 
		X = TestJacobian.X
		g = invtf.Generator() 
		g.add(invtf.layers.Inv1x1Conv(input_shape=X.shape[1:])) 
		g.compile()
		g.init(X[:100])
		g.fit(X[:1], verbose=False)
		self.assertJacobian(g, X)

	def test_resblock_init(self):

		X = TestJacobian.X
		g = invtf.Generator()

		# Create model
		rb  = invtf.approximation_layers.ResidualBlock(input_shape=X.shape[1:])
		params = {'strides': [1,1], 'padding': 'SAME'}

		def add_conv(kernels, input_shape):
			snc = invtf.approximation_layers.SpectralNormalizationConstraint(0.9, input_shape, **params)
			conv = Conv2D(3, kernel_size=kernels, input_shape=input_shape, kernel_constraint=snc, **params)
			rb.add(conv)
			return conv.compute_output_shape(input_shape)

		out_shape = X.shape
		for i in [3, 1, 3]: 
			out_shape = add_conv(i, out_shape)
		# End create model

		g.add(rb)
		g.init(X)
		print("Compile")
		g.compile()
		print("Fit")
		g.fit(X, batch_size=2)
		print("Done fit")
		print("AssertJacobian")
		self.assertJacobian(g, X)

	"""def test_glow_init(self): 
		X = tf.random.normal((1,32,32,3), 0, 1)
		g = invtf.Generator()tJacobian(g, X)

	def test_glow_fit(self): 
		X = tf.random.normal((1, 32,32,3), 0, 1)
		g = invtf.Generator()], verbose=False)
		self.assertJacobian(g, X)"""

	# Issue with GradientTape.Jacobian for FFT3D causes two below to break.
	"""
	def test_3dconv_init(self): 
		X = tf.random.normal((1, 32,32,3), 0, 1)
		g = invtf.Generator() 
		g.add(invtf.Conv3DCirc()) # initialize not like ones so it becomes zero. 
		g.compile()
		g.predict(X[:1])
		self.assertJacobian(g, X)

	def test_3dconv_fit(self): 
		X = tf.random.normal((1, 32,32,3), 0, 1)
		g = invtf.Generator() 
		g.add(invtf.Conv3DCirc()) # initialize not like ones so it becomes zero. 
		g.compile()
		g.fit(X[:1], verbose=False, epochs=1) 
		self.assertJacobian(g, X)"""



