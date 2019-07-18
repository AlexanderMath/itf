import unittest
import sys
sys.path.append("../")
import invtf
import invtf.layers
#from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt 
import warnings


class TestInverse(unittest.TestCase): 

	X = keras.datasets.cifar10.load_data()[0][0][:1] # a single cifar image. 

	def assertInverse(self, g, X): 
		"""
			Input: 		
				g:		Model which has call(X) return a tensor that depends on X. 
				X:		Test data. 
	
			Computes the reconstruction of X and compares with X. 

		""" 
		rec = g.rec(X)
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
		
		#print("\t", X.reshape(X.size)[entry], rec.reshape(rec.size)[entry], end="")

	def test_actnorm_init(self): 
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.ActNorm(input_shape=X.shape[1:])) 
		g.compile()

		g.init(X[:1]) # fails due to input initialization? 

		self.assertInverse(g, X)

	def test_actnorm_fit(self):
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.ActNorm(input_shape=X.shape[1:])) 
		g.compile()
		g.init(X[:1])
		g.fit(X[:1], verbose=False) 
		self.assertInverse(g, X)


	def test_natural_bijection(self): 
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Squeeze(input_shape=X.shape[1:])) 
		g.add(invtf.discrete_bijections.NaturalBijection()) 
		g.compile()
		enc = g.predict(X[:1])[0]

		self.assertInverse(g, X)

	def test_natural_bijection_twice(self): 
		X = TestInverse.X  # this is 8 bit, convert to 5 bit by divide by 2**3
		X = X // 2**3
		g = invtf.Generator() 
		g.add(invtf.layers.Squeeze(input_shape=X.shape[1:])) 
		g.add(invtf.discrete_bijections.NaturalBijection()) 
		g.add(invtf.discrete_bijections.NaturalBijection()) 
		g.compile()
		enc = g.predict(X[:1])[0]
		self.assertInverse(g, X)

	"""def test_natural_bijection_twice_with_bit_layer(self): 
		X = TestInverse.X  # this is 8 bit, convert to 5 bit by divide by 2**3
		g = invtf.Generator() 
		g.add(invtf.layers.Squeeze(input_shape=X.shape[1:])) 
		g.add(invtf.layers.ReduceNumBits(bits=5))
		g.add(invtf.discrete_bijections.NaturalBijection()) 
		g.add(invtf.discrete_bijections.NaturalBijection()) 
		g.compile()
		enc = g.predict(X[:1])[0]

		rec = g.rec(X)
		A = np.allclose(rec, X//2**3, atol=1, rtol=0.1) # assumes data is in bytes. 

		self.assertTrue(A)"""


	def test_additive_relu(self): 
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Squeeze(input_shape=X.shape[1:]))
		# they add / subtract on the same part of input and thus cancel each other out. 
		g.add(invtf.layers.AdditiveCouplingReLU(part=0, sign=+1, strategy=invtf.coupling_strategy.SplitChannelsStrategy())) 
		g.add(invtf.layers.AdditiveCouplingReLU(part=0, sign=-1, strategy=invtf.coupling_strategy.SplitChannelsStrategy())) 
		g.compile()
		enc = g.predict(X[:1])[0]
		self.assertInverse(g, X)

	def test_additive_relu_part1(self): 
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Squeeze(input_shape=X.shape[1:]))
		g.add(invtf.layers.AdditiveCouplingReLU(part=1, sign=+1, strategy=invtf.coupling_strategy.SplitChannelsStrategy())) 
		g.add(invtf.layers.AdditiveCouplingReLU(part=1, sign=-1, strategy=invtf.coupling_strategy.SplitChannelsStrategy())) 
		g.compile()
		enc = g.predict(X[:1])[0]
		self.assertInverse(g, X)


	"""def test_additive_relu_circ3d(self): 
		X = TestInverse.X  
		d = 32*32*3
		g = invtf.Generator(invtf.latent.Normal(d)) 
		g.add(keras.layers.InputLayer(input_shape=(32,32,3)))
		g.add(invtf.Squeeze())

		g.add(invtf.layers.AdditiveCouplingReLU(part=0, sign=+1, strategy=invtf.coupling_strategy.SplitChannelsStrategy()))
		g.add(invtf.layers.Conv3DCirc())
		g.add(invtf.layers.AdditiveCouplingReLU(part=0, sign=-1, strategy=invtf.coupling_strategy.SplitChannelsStrategy()))

		g.compile()
		enc = g.predict(X[:1])[0]
		self.assertInverse(g, X)"""





	def test_3dconv_init(self): 
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Conv3DCirc(input_shape=X.shape[1:])) 
		g.compile()
		g.predict(X[:1])
		self.assertInverse(g, X)

	def test_3dconv_fit(self): 
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Conv3DCirc(input_shape=X.shape[1:])) 
		g.compile()
		g.fit(X[:1], epochs=1, verbose=False) 
		self.assertInverse(g, X)


	def test_invconv_init(self):  
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Inv1x1Conv(input_shape=X.shape[1:])) 
		g.compile()
		g.predict(X[:1])
		self.assertInverse(g, X)

	def test_invconv_fit(self):  
		X = TestInverse.X 
		g = invtf.Generator() 
		g.add(invtf.layers.Inv1x1Conv(input_shape=X.shape[1:])) 
		g.compile()
		g.predict(X[:1])
		g.fit(X[:1], epochs=1, verbose=False)
		self.assertInverse(g, X)
		


