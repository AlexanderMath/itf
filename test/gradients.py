import unittest
import sys
sys.path.append("../")
import invtf.latent
# import invtf.layers
#from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from invtf import Generator as GenConst
from invtf.layers import *

#from invtf import *
#from invtf.layers import *
from invtf.coupling_layers import *
from tensorflow.keras.layers import ReLU, Dense, Flatten, Conv2D	

class GeneratorGradTest(GenConst):

	def prune(self,l): return [x for sublist in l for x in sublist if len(sublist)>0]

	def compute_gradients(self,X):
		x = self.call(X)        #I think putting this in context records all operations onto the tape, thereby destroying purpose of checkpointing...
		last_layer = self.layers[-1]
		d = np.prod(X.shape[1:])
		#Computing gradients of loss function wrt the last acticvation
		with tf.GradientTape() as tape:
			tape.watch(x)
			loss = self.loss(x, x)    #May have to change
		grads_combined = tape.gradient(loss,[x])
		dy = grads_combined[0]
		y = x
		#Computing gradients for each layer
		gradients = []
		for layer in self.layers[::-1]:     
			if isinstance(layer, keras.layers.InputLayer): break
			x = layer.call_inv(y)
			dy,grads = layer.compute_gradients(x,dy,layer.log_det,d*np.log(2.))	#TODO implement scaling here -- DONE
			gradients=[grads]+gradients
			y = x 
		return self.prune(gradients)

	def actual_gradients(self,X):
		with tf.GradientTape() as tape:
			pred = self.call(X)
			loss = self.loss(pred, pred)
		grads = tape.gradient(loss,self.trainable_variables)
		return grads

class TestGradients(unittest.TestCase):
	X = tf.constant(keras.datasets.cifar10.load_data()[0][0][:1000].astype('f')) # a single cifar image batch.

	def assertGrad(self,g,X):
		computed_grads = g.compute_gradients(X)
		actual_grads = g.actual_gradients(X) 
		A = [np.allclose(np.abs(x[0]-x[1]),0,atol=10**(-4), rtol=0.1) for x in zip(computed_grads,actual_grads) if x[0] is not None]
		# print("computed",computed_grads,"actual_grads",actual_grads)
		print("Max discrepancy in gradients",np.max(np.array([np.max((np.abs(x[0]-x[1]))) for x in zip(computed_grads,actual_grads) if x[0] is not None])))
		self.assertTrue(np.array(A).all())

	def test_circ_conv_init(self):
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Conv3DCirc(input_shape=X.shape[1:]))
		g.predict(X[:1])
		self.assertGrad(g,X)		

	def test_circ_conv_fit(self):
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Conv3DCirc(input_shape=X.shape[1:]))
		g.compile()

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		

	def test_squeeze_circ_conv(self):
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Squeeze(input_shape=X.shape[1:]))
		g.add(Conv3DCirc())
		g.compile()

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		

	def test_squeeze_normalize_circ_conv(self):
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Squeeze(input_shape=X.shape[1:]))
		g.add(Normalize())
		g.add(Conv3DCirc())
		g.compile()

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		


	def test_add_relu(self): 
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Squeeze(input_shape=X.shape[1:]))
		g.add(AdditiveCouplingReLU())
		g.compile()

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		


	def test_add_relu_circ_conv(self): 
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Squeeze(input_shape=X.shape[1:]))
		for _ in range(5): 
			g.add(Conv3DCirc())
			g.add(AdditiveCouplingReLU())
			
		g.add(Conv3DCirc())
		g.compile()

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		

		


	def test_deep_circ_conv(self):
		X = TestGradients.X 

		g = GeneratorGradTest() 
		g.add(Squeeze(input_shape=X.shape[1:]))
		g.add(Normalize())
		for _ in range(100): 
			g.add(Conv3DCirc())

		g.compile()

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		



	# pass 
	def test_act_norm_init(self):
		X = TestGradients.X 
		g = GeneratorGradTest() 
		g.add(ActNorm(input_shape=X.shape[1:]))
		g.compile()

		self.assertGrad(g,X)		

	# pass 
	def test_act_norm_fit(self):
		X = TestGradients.X 
		g = GeneratorGradTest() 
		g.add(ActNorm(input_shape=X.shape[1:]))
		g.compile() 

		g.fit(X[:1], verbose=0, memory="constant")
		self.assertGrad(g,X)		







	# dies, but don't focus on this for now. 
	"""def test_inv_conv(self):
		X = TestGradients.X 
		d = 32*32*3
		g = GeneratorGradTest(invtf.latent.Normal(d)) 
		g.add(Inv1x1ConvPLU())
		g.predict(X[:1])
		self.assertGrad(g,X)		

	def test_affine_coupling(self):
		X = np.random.normal(0,1,(5,2,2,2)).astype('f')
		print(X.shape)
		d = 2*2*2
		g = GeneratorGradTest(invtf.latent.Normal(d)) 
		b = AffineCoupling()
		b.add(Flatten())
		b.add(Dense(d,activation='sigmoid'))
		g.add(Squeeze())
		g.add(Conv3DCirc())
		g.add(b)
		g.add(Conv3DCirc())
		# g.predict(X[:1])
		self.assertGrad(g,X)		"""

