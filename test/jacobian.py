
import unittest
import sys
sys.path.append("../")
import invtf
import invtf.layers
import tensorflow as tf
import tensorflow.keras as keras
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

		with tf.GradientTape(persistent=True) as t: 
			t.watch(X)
			z = g.call(X)

		J = t.jacobian(z, X, experimental_use_pfor=False)

		J = tf.reshape(J, (32*32*3, 32*32*3))

		lgdet1 = tf.math.log(tf.linalg.det(J)).numpy()
		lgdet2 = g.log_det().numpy()

		#print(lgdet1, lgdet2)
		
		# If the following equation is element-wise True, then allclose returns True.
		# 		absolute(a - b) <= (atol + rtol * absolute(b))
		# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
		A = np.allclose( lgdet1, lgdet2 , atol=10**(-3), rtol=0.15)  # deciding these values are very difficult for all tests.. 

		self.assertTrue(A)
		#print("\t", lgdet1, lgdet2, "\t", end="")
		

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



