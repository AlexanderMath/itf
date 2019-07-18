import invtf 				
from invtf.visualize 	import visualize_training_2D
from invtf.layers 		import *
from invtf.dequantize 	import * 
from tensorflow			import keras
from tensorflow.keras 	import Sequential

from tensorflow.keras.layers import ReLU, Dense, Flatten, Reshape, Conv2D
from tensorflow.keras.models import Sequential

import invtf.latent as latent


class PredefinedModel(): 

	"""
		Have example with use case similar to Generator class. 
	"""



class NICE(): 

	"""
		Learn MNIST using (something close to) NICE architecture. 
		They get around 4.36

		Experiments: 
			- compare normal vs logistic distribution. 
			- try zero initialization. 
			- compare with conv networks instead of dense. 
			- they normalize to [0, 1], try [-1, +1] instead. 
			  Furthermore, try identity initialization and chose interval to "fit" gaussian. 

			(note: they train for 1500 epochs and cross validate). 

		Comments:
			Issue 1: 
			In [1] they write "... we add a uniform noise of 1/256 to the data and rescale it to be in [0,1] after dequantization. "
			It is not clear if they normalize to [0,1] first and then add 1/256, or they add 1/256 to the {0,1,2,...,2**8-1}. 
			I added U(0,1) to {0,1,2,...,2**8-1} as outlined in the flow++ article. 

			Currently not sure how they have output layer of all coupling networks; do they just have d//2 linear output neurons? 


		Lacking information:
			- No batch size is reported. 
			- No information on weight initialization. 
			- No pseudo-code. 


		Differences:
			- Potentially the dequantization (see comments above) 
			- The scaling in the end has bias which they do not have. 

		Current issues: 
			- Training yields NAN after a few epochs, besides that everything seems to be working.  Get tensorbaord to work and start debugging. It seems this usually happens around when the following two gets close: 
					lg_latent: 2.7125 - lg_perfect: 2.8825 
		

	"""

	def model(X):   # assumes mnist. 
		n, d = X.shape

		# This differs from MNIST to CIFAR in original article. 
		#g = invtf.Generator(latent.Logistic(d))  
		g = invtf.Generator(latent.Normal(d))  

		# Pre-process steps. 
		g.add(UniformDequantize	(input_shape=[d])) 
		g.add(Normalize			(input_shape=[d]))

		# Build model using additive coupling layers. 
		for i in range(0, 4): 

			ac = AdditiveCoupling(part=i%2, strategy=EvenOddStrategy())
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros")) 
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(1000, activation="relu", bias_initializer="zeros", kernel_initializer="zeros"))
			ac.add(Dense(d//2, bias_initializer="zeros")) 

			g.add(ac) 

		g.add(Affine(exp=True))

		g.compile(optimizer=keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.01, epsilon=10**(-4)))

		g.init(X[:100])

		return g



class RealNVP(): 

	"""
		1) Change from additive to affine coupling layer: DONE. 
			It seems log computations are fine, but there might be a few issues. 
			!!! The permutation is very simple compared to channel-wise masking / spatial checkerboard.
			TODO: implement channel-wise masking + spatial checkerboard. 

		3) implement squeeze (try different ones): DONE
		
		4) Implement multi-scale architecture. : DONE 

		2) Change from dense to residual conv net with ??bnorm???: 

		Differences: 
			The implementation follows GLOW, so there is not exp in the scaling. 

		Further development: 
		- Do progressive kind of training? 

	"""

	def model(X):  
		input_shape = X.shape[1:]
		d 			= np.prod(input_shape)

		g = invtf.Generator(latent.Normal(d)) 

		# Pre-process steps. 
		g.add(UniformDequantize	(input_shape=input_shape)) 
		g.add(Normalize			(input_shape=input_shape))

		# Build model using additive coupling layers. 
		g.add(Squeeze())

		strategy = SplitChannelsStrategy()

		for i in range(0, 2): 
			for j in range(2): 

				ac = AffineCoupling(part=j%2, strategy=strategy)
				ac.add(Flatten())
				ac.add(Dense(100, activation="relu"))
				ac.add(Dense(100, activation="relu"))
				ac.add(Dense(100, activation="relu"))
				ac.add(Dense(d, bias_initializer="ones", kernel_initializer="zeros"))

				g.add(ac) 
			
			g.add(Squeeze())

			g.add(MultiScale()) # adds directly to output. For simplicity just add half of channels. 
			d = d//2

		ac = AffineCoupling(part=j%2, strategy=strategy)
		ac.add(Flatten())
		ac.add(Dense(100, activation="relu"))
		ac.add(Dense(100, activation="relu"))
		ac.add(Dense(100, activation="relu"))
		ac.add(Dense(d, bias_initializer="ones", kernel_initializer="zeros"))

		g.add(ac) 

		g.compile(optimizer=keras.optimizers.Adam(0.0001))

		g.init(X[:1000])

		ac.summary()

		return g

class Conv3D(): 

	"""
			Currently only utilize around 10 % of GPU with batchsize 32.
			with batchsize 512 it is around 30. TODO: Fix that.

	""" 

	def model(X, verbose=False):  


		input_shape = X.shape[1:]

		g = invtf.Generator() 

		# seems to work, but it is a bit unstable. 
		g.add(Squeeze(input_shape=input_shape))
		g.add(UniformDequantize	()) 
		g.add(Normalize			())  

		for j in range(10):  
			g.add(ActNorm())
			g.add(Conv3DCirc())  
			g.add(AdditiveCouplingReLU(part=0, sign=+1))

			g.add(ActNorm())
			g.add(Conv3DCirc()) 
			g.add(AdditiveCouplingReLU(part=0, sign=-1))

			g.add(ActNorm())
			g.add(Conv3DCirc())  
			g.add(AdditiveCouplingReLU(part=1, sign=+1))

			g.add(ActNorm())
			g.add(Conv3DCirc()) 
			g.add(AdditiveCouplingReLU(part=1, sign=-1))


		g.compile()
		g.init(X[:1000])

		if verbose: 
			for layer in g.layers: 
				if isinstance(layer, AffineCoupling): layer.summary()

			for layer in g.layers:
				if isinstance(layer, VariationalDequantize): layer.summary()

		return g


class Glow(): 

	"""
		Curiously they don't use residual networks in coupling layers as RealNVP does. 

	""" 

	def model(X, verbose=False):  

		# Glow details can be found at : https://github.com/openai/glow/blob/master/model.py#L376
		default_initializer = keras.initializers.RandomNormal(stddev=0.05)
		width 				= 128 # width in glow is 512 but we lowered for speed; how much does this hurt performance? 
		c 					= X.shape[-1]

		input_shape = X.shape[1:]
		d 			= np.prod(input_shape)

		g = invtf.Generator(latent.Normal(d)) 

		# Pre-process steps. 
		g.add(keras.layers.InputLayer(input_shape=input_shape))
		g.add(UniformDequantize	(input_shape=input_shape)) 
		g.add(Normalize			(input_shape=input_shape)) 

		# Build model using additive coupling layers. 
		g.add(Squeeze())
		c = 4*c

		for i in range(0, 2): 

			for j in range(1): 

				g.add(ActNorm())
				#g.add(Inv1x1Conv()) 

				ac = AffineCoupling(part=j%2, strategy=SplitChannelsStrategy())
		
				ac.add(Conv2D(width, kernel_size=(3,3), activation="relu", padding="SAME", 
								kernel_initializer=default_initializer, bias_initializer="zeros")) 
				ac.add(Conv2D(width, kernel_size=(1,1), activation="relu", padding="SAME", 
								kernel_initializer=default_initializer, bias_initializer="zeros"))
				ac.add(Conv2D(c, kernel_size=(3,3), 				   padding="SAME",
								kernel_initializer="zeros", bias_initializer="ones"))  # they add 2 here and apply sigmoid. 
				

				g.add(ac) 
			
			#g.add(Squeeze())
			#c = c * 4

			#g.add(MultiScale()) # adds directly to output. For simplicity just add half of channels. 
			#d = d//2

		g.compile(optimizer=keras.optimizers.Adam(0.001))

		g.init(X[:1000]) # initialize actnorm data dependently. 

		if verbose: 
			for layer in g.layers: 
				if isinstance(layer, AffineCoupling): layer.summary()

			for layer in g.layers:
				if isinstance(layer, VariationalDequantize): layer.summary()

		return g



class ConvInvResNet(): 

	
	def model(X):

		# instead of 3D use ND and see how different dimension change performance. 
		# also, with resnet block, have maybe 
		# 		y = sum_j conv_(jD) conv(x)

		for _ in range(10000): # use O(1) memory trick. 
			g.add(ResidualConv3DCirc()) # can only contract 
			g.add(Conv3DCirc()) # is allowed to expand. 


class FlowPP():  # flow++ ;; - attention and logit coupling layers. 

	def model(X, verbose=False):  

		default_initializer = keras.initializers.RandomNormal(stddev=0.05)
		width 				= 128 
		c = 12


		input_shape = X.shape[1:]
		d 			= np.prod(input_shape)

		g = invtf.Generator(latent.Normal(d)) 

		# Pre-process steps. 
		#g.add(UniformDequantize	(input_shape=input_shape)) 

		g.add(keras.layers.InputLayer(input_shape=input_shape))

		# Build model 
		g.add(Squeeze())

		vardeq = VariationalDequantize() 

		for j in range(2): 
			ac = AffineCoupling(part=j%2, strategy=SplitChannelsStrategy())
			ac.add(Conv2D(width, kernel_size=(3,3), activation="relu", padding="SAME", 
							kernel_initializer=default_initializer, bias_initializer="zeros")) 
			ac.add(Conv2D(width, kernel_size=(1,1), activation="relu", padding="SAME", 
							kernel_initializer=default_initializer, bias_initializer="zeros"))
			ac.add(Conv2D(c, kernel_size=(3,3), 				   padding="SAME",
							kernel_initializer="zeros", bias_initializer="ones"))  # they add 2 here and apply sigmoid. 

			vardeq.add(ActNorm())
			#vardeq.add(ac)  # this loss starts being 8 or something crazy... 

		g.add(vardeq) # print all loss constituents of this? 

		g.add(Normalize		(input_shape=input_shape)) 

		for i in range(0, 2): 

			for j in range(2): 

				g.add(ActNorm())
				#g.add(Conv3DCirc())
				#g.add(Inv1x1Conv()) 

				ac = AffineCoupling(part=j%2, strategy=SplitChannelsStrategy())
		
				ac.add(Conv2D(width, kernel_size=(3,3), activation="relu", padding="SAME", 
								kernel_initializer=default_initializer, bias_initializer="zeros")) 
				ac.add(Conv2D(width, kernel_size=(1,1), activation="relu", padding="SAME", 
								kernel_initializer=default_initializer, bias_initializer="zeros"))
				ac.add(Conv2D(c, kernel_size=(3,3), 				   padding="SAME",
								kernel_initializer="zeros", bias_initializer="ones"))  # they add 2 here and apply sigmoid. 
				

				#g.add(Inv1x1Conv())
				g.add(ac) 
			
			#g.add(Squeeze())
			#c = c * 4

			#g.add(MultiScale()) # adds directly to output. For simplicity just add half of channels. 
			#d = d//2

		g.compile(optimizer=keras.optimizers.Adam(0.001))

		g.init_actnorm(X[:1000]) # how much does this change loss? 

		if verbose: 
			for layer in g.layers: 
				if isinstance(layer, AffineCoupling): layer.summary()

			for layer in g.layers:
				if isinstance(layer, VariationalDequantize): layer.summary()

		return g



