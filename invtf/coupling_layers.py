import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.layers import ReLU
from invtf.override import print_summary
from invtf.coupling_strategy import *
from invtf.coupling_layers 	import *

class CouplingLayer(keras.layers.Layer): 
	"""
		Coupling layers [1,2,3] can make an invertible variant of an arbitrary function 'm' 
		with a tractable jacobian log determinant. In this way they are very flexible, since they 
		assume nothing on m, it could e.g. be a deep convolution neural network. The main 
		disadvantage of coupling layers is that 'm' is constrained to only see one part of 
		the input and change the other part. 

		To this end, [2] introduced a type of 1x1 convolution which can be used in combination
		with coupling layers. This type of convolution generalize permutations, allowing the
		network to encode the most important information into the part 'm' sees. 

		Types:
			Additive Coupling Layers: [1]	(implemented)
			Affine Coupling Layers: [4] 	(implemented)
			Continuous Mixture CDFS [3]. 

		Benchmark:
			In [5] we benchmark several recent models with different types of coupling layers. 
			In this we demonstrate trade-offs between generative performance (as measured by
			negative log likelihood) and training time, across several dataset. 

			Based on this benchmark we recommend using X when Y, and Z when T. 


		[1] NICE: Non-linear Independent Components Estimation		https://arxiv.org/abs/1410.8516
		[2] Glow: Generative Flow with Invertible 1x1 Convolutions 	https://arxiv.org/abs/1807.03039
		[3]	Flow++: Improving Flow-Based Generative Models with 	https://arxiv.org/abs/1902.00275
			Variational Dequantization and Architecture Design 
		[4] Density estimation using Real NVP						https://arxiv.org/abs/1605.08803
		[5]	https://github.com/AlexanderMath/InvTF/benchmark/coupling.ipynb
	"""
	def call(self, X): 		raise NotImplementedError()
	def call_inv(self, X): 	raise NotImplementedError()
	def log_det(self, X): 	raise NotImplementedError()
	def add(self, layer): 	raise NotImplementedError()

class AdditiveCoupling(CouplingLayer):  
	"""
		Given a function 'm' the additive coupling splits the input x=(x1 x2) and computes
	
			     [y1]   [x1 + 0    ] 						[y1		 ]	 [x1]
			y  = [y2] = [x2 + m(x1)]	with inverse 	x = [y2-m(y1)] = [x2]

		Furthermore, if x1 and x2 are equally large, the log jacobian determinant is zero,	
		please see [1] for details. The function 'm' can be constructed using the 
		Sequential keras API, see an example below. 

		Example (1): (image data of size [n x h x w x c]). 
			
			> import invtf
			> import tensorflow.keras as keras

			> # Load data
			> X = invtf.datasets.cifar10()
			> input_shape = X.shape[1:]

			> # Model
			> g = invtf.Generator()
			> g.add(invtf.dequantize.UniformDequantize(input_shape=input_shape)) 
			> g.add(invtf.layers.Normalize()) 
			> g.add(invtf.layers.Squeeze())

			> c=3*4 # refactor this away. 

			> for i in range(4): 

			> 	ac = invtf.coupling_layers.AdditiveCoupling(part=i%2, strategy=invtf.coupling_strategy.SplitChannelsStrategy())
			> 	ac.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding="SAME"))
			> 	ac.add(keras.layers.Conv2D(c//2, kernel_size=(3,3), padding="SAME"))
			> 	g.add(ac)

			> 	# if i == 2: g.add(invtf.layers.MultiScale()) 

			> # Prepare model for training and print summary. 
			> g.compile()  
			> g.init(X[:1000])  
			> g.summary()

			> # Train model. 
			> g.fit(X, batch_size=512)


		Example (2): (vectorized data of size [n x d] as in [1]). 

			> import invtf
			> import tensorflow.keras as keras

			> # Load data
			> X = invtf.datasets.cifar10()
			> X = X.reshape(X.shape[0], -1)
			> input_shape = X.shape[1:]

			> # Model
			> g = invtf.Generator()
			> g.add(invtf.dequantize.UniformDequantize(input_shape=input_shape)) 
			> g.add(invtf.layers.Normalize()) 

			> for i in range(4): 
			> 	ac = invtf.coupling_layers.AdditiveCoupling(part=i%2, strategy=invtf.coupling_strategy.SplitOnHalfStrategy())
			> 	ac.add(keras.layers.Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="normal")) 
			> 	ac.add(keras.layers.Dense(input_shape[0]//2))
			> 	g.add(ac)

			>	# Removed for now, multi-scale assumes (n, h, w, c) shape. 
			> 	# if i == 2: g.add(invtf.layers.MultiScale()) 

			> # Prepare model for training and print summary. 
			> g.compile()  
			> g.init(X[:1000])  
			> g.summary()

			> # Train model. 
			> g.fit(X, batch_size=512)


		Comments:

			It would be interesting to see combinations of Additive/Affine coupling layers. 
			Even though Affine generalizes Additive the loss landscape is probably very different. 

		TODO:
		
			(1) Make sure multi-scale architecture also works. 
			(2) Automatically compute the size of channels. 
			(3) Write tests for computation of sizes etc. 
			(4) Default to different strategies depending on vectorized input or not. 
			Maybe have strategy implement both vectorized and (hxwxc) shaped input. 
			(5) Do everything done for additive coupling also for Affine coupling. 
	
	"""

	unique_id = 1

	def __init__(self, part=0, strategy=()): 
		super(AdditiveCoupling, self).__init__(name="add_coupling_%i"%AdditiveCoupling.unique_id)
		AdditiveCoupling.unique_id += 1

		self.part 		= part 
		self.strategy 	= strategy
		self.layers 	= []

	def add(self, layer): self.layers.append(layer)

	def build(self, input_shape):
		print("Input Shape: ", input_shape)

		if len(input_shape) == 2: # input is vectorized. 
			_, d = input_shape 

			self.layers[0].build(input_shape=(None, d//2))
			out_dim = self.layers[0].compute_output_shape(input_shape=(None, d//2))

		elif len(input_shape) == 4: # assumes shape (n, h, w, c) used e.g. by images. 
			n, h, w, c = input_shape 
			print(n, h, w, c)

			self.layers[0].build(input_shape=(None, h, w, c//2))
			out_dim = self.layers[0].compute_output_shape(input_shape=(None, h, w, c//2))
			print(self.layers[0])
			print(out_dim)
			#self.layers[0].output_shape_ = out_dim # refactor this away? used in multi-scale, but I think we can circumvent. 

		for layer in self.layers[1:]:  
			print(out_dim)
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)
			print(out_dim)

		super(AdditiveCoupling, self).build(input_shape)
		self.built = True

	def call_(self, X): 
		for layer in self.layers: 
			X = layer.call(X)
		return X

	def call(self, X): 		
		shape 	= tf.shape(X)
		print(shape)

		x0, x1 	= self.strategy.split(X)
		print(x0.shape, x1.shape)

		if self.part == 0: x0 		= x0 + self.call_(x1)
		if self.part == 1: x1 		= x1 + self.call_(x0)

		X 		= self.strategy.combine(x0, x1)
		print(X.shape)

		X 		= tf.reshape(X, shape)
		print(X.shape)
		return X

	def call_inv(self, Z):	 
		shape 	= tf.shape(Z)

		z0, z1 	= self.strategy.split(Z)
		
		if self.part == 0: z0 		= z0 - self.call_(z1)
		if self.part == 1: z1 		= z1 - self.call_(z0)

		Z 		= self.strategy.combine(z0, z1)

		Z 		= tf.reshape(Z, shape)
		return Z


	def log_det(self): 		return 0. 

	def compute_output_shape(self, input_shape): return input_shape




class AffineCoupling(keras.layers.Layer): 
	"""
			

	The affine coupling layer is described in NICE, REALNVP and GLOW. 
	The description in Glow use a single network to output scale s and transform t, 
	it seems the description in REALNVP is a bit more general refering to s and t as 
	different functions. From this perspective Glow change the affine layer to have
	weight sharing between s and t. 
	 Specifying a single function is a lot simpler code-wise, we thus use that approach. 


	For now assumes the use of convolutions 



	"""



	def add(self, layer): self.layers.append(layer)

	unique_id = 1

	def __init__(self, part=0, strategy=SplitChannelsStrategy()): 
		super(AffineCoupling, self).__init__(name="aff_coupling_%i"%AffineCoupling.unique_id)
		AffineCoupling.unique_id += 1
		self.part 		= part 
		self.strategy 	= strategy
		self.layers = []
		self._is_graph_network = False
		self.precomputed_log_det = 0.

	def _check_trainable_weights_consistency(self): return True

	def build(self, input_shape):

		# handle the issue with each network output something larger. 
		_, h, w, c = input_shape


		h, w, c = self.strategy.coupling_shape(input_shape=(h,w,c))

		self.layers[0].build(input_shape=(None, h, w, c))
		out_dim = self.layers[0].compute_output_shape(input_shape=(None, h, w, c))
		self.layers[0].output_shape_ = out_dim

		for layer in self.layers[1:]:  
			layer.build(input_shape=out_dim)
			out_dim = layer.compute_output_shape(input_shape=out_dim)
			layer.output_shape_ = out_dim


		super(AffineCoupling, self).build(input_shape)
		self.built = True

	def call_(self, X): 

		in_shape = tf.shape(X)
		n, h, w, c = X.shape

		for layer in self.layers: 
			X = layer.call(X) 

		# TODO: Could have a part of network learned specifically for s,t to not ONLY have wegith sharing? 
		
		# Using strategy from 
		# https://github.com/openai/glow/blob/eaff2177693a5d84a1cf8ae19e8e0441715b82f8/model.py#L376
		X = tf.reshape(X, (-1, h, w, c*2))
		s = X[:, :, :, ::2] # add a strategy pattern to decide how the output is split. 
		t = X[:, :, :, 1::2]  
		#s = tf.math.sigmoid(s)

		#s = X[:, :, w//2:, :]
		#t = X[:, :, :w//2, :]  

		s = tf.reshape(s, in_shape)
		t = tf.reshape(t, in_shape)

		return s, t

	def call(self, X): 		

		x0, x1 = self.strategy.split(X)

		if self.part == 0: 
			s, t 	= self.call_(x1)
			x0 		= x0*s + t # glow changed order of this? i.e. translate then scale. 

		if self.part == 1: 
			s, t 	= self.call_(x0)
			x1 		= x1*s + t 

		self.precompute_log_det(s, X)

		X 		= self.strategy.combine(x0, x1)
		return X

	def call_inv(self, Z):	 
		z0, z1 = self.strategy.split(Z)
		
		if self.part == 0: 
			s, t 	= self.call_(z1)
			z0 		= (z0 - t)/s
		if self.part == 1: 
			s, t 	= self.call_(z0)
			z1 		= (z1 - t)/s

		Z 		= self.strategy.combine(z0, z1)
		return Z

	def precompute_log_det(self, s, X): 
		n 		= tf.dtypes.cast(tf.shape(X)[0], tf.float32)
		self.precomputed_log_det = tf.reduce_sum(tf.math.log(tf.abs(s))) / n

	def log_det(self): 		  return self.precomputed_log_det

	def compute_output_shape(self, input_shape): return input_shape

	def summary(self, line_length=None, positions=None, print_fn=None):
		print_summary(self, line_length=line_length, positions=positions, print_fn=print_fn) # fixes stupid issue.




