import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'

__version__ = "0.0.1"

import tensorflow as tf

import invtf.coupling_strategy
import invtf.datasets
import invtf.dequantize
import invtf.discrete_bijections
import invtf.grow_memory
import invtf.latent
import invtf.layers
import invtf.models
import invtf.override
import invtf.visualize

import tensorflow.keras as keras 
import numpy as np
import matplotlib.pyplot as plt 

print("TF Version: \t", tf.__version__)
print("Eager: \t\t", tf.executing_eagerly())
print("InvTF Version: \t", __version__)
print("-------------------------------\n")


class Generator(keras.Model): 
	"""
		Linear stack of invertible layers for a generative model. 


		----------------------------------------------------------------------------------------

		Example:	(TODO: Add link to Google Colab w/ 'simple.py' and "pip install invtf". )

			> import invtf
			> import tensorflow.keras as keras

			> # Load data
			> X = invtf.datasets.cifar10()
			> input_shape = X.shape[1:]

			> # Model
			> g = invtf.Generator()

			> # Pre-process
			> g.add(invtf.dequantize.UniformDequantize(input_shape=input_shape)) 
			> g.add(invtf.layers.Normalize()) 

			> # Actual model. 
			> g.add(invtf.layers.Squeeze())

			> for _ in range(10): 
			> 	g.add(invtf.layers.ActNorm())
			> 	g.add(invtf.layers.Conv3DCirc())
			> 	g.add(invtf.layers.AdditiveCouplingReLU()) 
			>    
			>	# If the line below is uncommented, the architecture
			>	# becomes a multi-scale architecture similar to [1,2].
			>	# if i == 5: g.add(invtf.layers.MultiScale())

			> # Prepare model for training and print summary. 
			> g.compile()		# handles maximum likelihood computations. 
			> g.init(X[:1000])	# handles data dependent initialization as used in [2]. 
			> g.summary()

			> g.fit(X)

		It is now possible to use the training model to sample or interpolate between
		training examples. 

		----------------------------------------------------------------------------------------

		Comments:

			(1) The Generator class supports an API similar to that of Keras Sequential [4]. 
			The example above demonstrates how to build a invertible generative model with
			this API. After defining a model, calling "model.fit" automatically trains the model 
			to maximize the likelihood of the data. Notably, the user does not need to handle any 
			likelihood computations, these are handled automatically by the framework. 

			TODO: outline how one writes new layers by implementing call, call_inv and log_det. 

			Even though the Generator class has an API similar to Sequential, it inherits 
			from the Model class. This is caused by the multi-scale architecture. Essentially,
			Sequential only allows all layers to have one output, and the layer that implements
			the multi-sclae architecture has two outputs. 


			(2) Previously, dequantization and data normalization was thought as pre-processing 
			done before the data is given to the model. Both are treated as a part of the 
			model for the following reasons: 

				- Normalization affects the loss function of invertible generative models,
				  in fact, the normalization typically has a very large jacobian determinant. 
				  Having normalization as a part of the function model allows us to 
				  automatically update the loss function. 

				- The way we choose to dequantize discrete data is a modeling assumption. 
				  We highly recommend reading [3] on variational dequantization. 

			We chose Dequantization and Normalization to be Keras Layers for convenience. 

			(3) Implementing the multi-scale architecture. Automatically handling these 
			computations are a bit tedious, and, unfortunately, complicates reading the code. 

		----------------------------------------------------------------------------------------

		Known-Issues:

			(1) Each multi-scale level increases loss substantially. Currently, it is not clear
			if this is caused by bad initialization or an error in the likelihood computations. 

		----------------------------------------------------------------------------------------

		References:

			[1] Density Estimation using Real NVP. 							https://arxiv.org/abs/1605.08803
			[2] Glow: Generative Flow with Invertible 1x1 Convolutions. 	https://arxiv.org/abs/1807.03039
			[3] Flow++: Improving Flow-Based Generative Models with  		https://arxiv.org/abs/1902.00275
				Variational Dequantization and Architecture Design
			[4] The Sequential model API									https://keras.io/models/sequential/


		Todo:
			

	"""

	def __init__(self, latent=latent.Normal()):
		"""
			
		"""
		super(Generator, self).__init__()
		self.latent = latent 

	def predict(self, X, dequantize=True): 
		"""
			Computes the encoding under the generative model of the input samples. 

			Arguments: 

				X:				Input samples, typically Numpy array.  

				dequantize: 	Boolean. This allows the user to disable dequantization. This can be useful 
								when testing the numerical stability of inverse computations, since the 
								dequantization has no inverse. 

			Returns:		
				Numpy array X of encoding and a list Zs containing numpy arrays of intermediate 
				multi-scale outputs. 

		"""
		Zs = [] 

		for layer in self.layers: 

			# Allow deactivating de-quantize. 
			if not dequantize and isinstance(layer, invtf.dequantize.Dequantize): 		continue	

			# Handle multi-scale intermediate output. 
			if isinstance(layer, invtf.layers.MultiScale): 
				X, Z = layer.call(X)
				Zs.append(Z)
				continue

			X = layer.call(X)

		return X, Zs

	def predict_inv(self, X, Zs=None): 
		"""
			Reconstructs encodings (X, Zs) under the generative model. 

			Arguments: 

				X, Zs:			NumPy arrays with different levels of encoding. 

			Returns:		
				Numpy array X of reconstructions. 

		"""
		n = X.shape[0]

		for layer in self.layers[::-1]: 
			if isinstance(layer, keras.layers.InputLayer): continue

			if isinstance(layer, invtf.layers.MultiScale): 
				X = layer.call_inv(X, Zs.pop())

			else: 
				X = layer.call_inv(X)

		return np.array(X)



	def loss(self, y_true, y_pred):	 
		"""
			Computes negative log likelihood in bits per dimension. If the model uses Variational 
			Dequantization it incorporates this into the loss function, see equation (12) from [1]. 

			Arguments: 

				y_true:			Dummy variable. It is required by Keras API, see [2], but the 
								likelihood computations does not use it.
				y_pred:			Output of last layer (encoding of model). 

			Returns:		
				The negative log likelihood of the model on data X such that encodings y_pred 
				is the output of predict(X). 


			Comments: 

				It often useful to inspect the different components of the loss function, for example,
				how big is the jacobian term compared to the latent density term. To facilitate this,
				each loss term is computed by another function which by default is printed during 
				training. To make sure the different terms are comparable normalization is handled
				inside each function. 

			
			[1] Flow++: Improving Flow-Based Generative Models with 			https://arxiv.org/pdf/1902.00275
				Variational Dequantization and Architecture Design
			[2] Usage of loss functions											https://keras.io/losses/

		"""
		return self.loss_log_det(y_true, y_pred) + self.loss_log_latent_density(y_true, y_pred) + self.loss_log_var_dequant(y_true, y_pred)


	def compile(self, optimizer=keras.optimizers.Adam(0.001), **kwargs):	
		"""
			Compiles the model to minimize negative log likelihood.  The different terms of the loss 
			function is added as metric, this is often useful information during training. 

			Comments:
				The multi-scale layer splits input into two outputs, one is passed on as for 
				subsequent forward pass, the other part is passed directly to output. Currently,
				this is done by using a list. The first element is used for subsequent computation
				and all other elements must later be added together to form output. This happens
				at the self.outputs = [tf.concat(..)] line. 

		"""

		if "loss" in kwargs.keys(): raise Exception("Currently only supports training with maximum likelihood. Please leave loss unspecified, e.g. 'model.compile()'. ")

		# In the case of Multi-Scale architecture, there will be multiple outputs. 
		# The computations inside the loss function assumes there is just a one output. 
		# This is handled by flattening, combining and finally reshaping the outputs 
		# into one single output. In the simple case the shape of the final output is 
		# that of the input; a more complicated case happens when the model has Discrete 
		# Bijections, in this case, it reshapes to the output shape of the last Discrete Bijection. 
		# For now it is assumed NaturalBijection is the only Discrete Bijection. TODO: refactor with abstract class. 
		self.outputs = [tf.reshape(output, (tf.shape(output)[0], np.prod(output.shape[1:]))) for output in self.outputs]
		self.outputs = tf.concat(self.outputs, axis=-1) # concatenated. 

		# In the end reshape to fit input last discrete bijection. If none fit input shape
		h, w, c = self.input_shape[1:]
		for layer in self.layers: 
			if isinstance(layer, invtf.discrete_bijections.NaturalBijection): 
				h, w, c = layer.output_shape[1:]

		self.outputs 		= [tf.reshape(self.outputs, (tf.shape(self.outputs)[0], h, w, c))] 
		self.output_names 	= [self.output_names[-1]] 

		kwargs['optimizer'] = optimizer
		kwargs['loss'] 		= self.loss 

		def lg_det(y_true, y_pred): 	return self.loss_log_det(y_true, y_pred)
		def lg_latent(y_true, y_pred): 	return self.loss_log_latent_density(y_true, y_pred)
		def lg_perfect(y_true, y_pred): return self.loss_log_latent_density(y_true, self.latent.sample(shape=tf.shape(y_pred))) 
		def lg_vardeqloss(y_true, y_pred): return self.loss_log_var_dequant(y_true, y_pred)

		kwargs['metrics'] = [lg_det , lg_latent, lg_perfect, lg_vardeqloss]

		super(Generator, self).compile(**kwargs)


	def loss_log_det(self, y_true, y_pred): 
		"""
			Computes negative log determinant, the first term of the loss function. 
			The loss is normalized to be in bits per dimension.

			Arguments:
				y_true/y_pred: 	Dummy variables. The Keras API requires both arguments 
								for a function to be a loss/metric, however, the log 
								determinant does not depend on them. 

		"""
		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), tf.float32)
		norm		= d * np.log(2.) 

		# Divide by /d to get per dimension. 
		# Divide by log(2) to go from log base E (default in tensorflow) to log base 2. 
		# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math/log
		log_det 	= self.log_det() / norm

		return 		- log_det  


	def loss_log_latent_density(self, y_true, y_pred): 
		"""
			Computes log density of encoded data in latent space, the second term of the 
			loss function. The loss is normalized to be in bits per dimension, averaged
			over the number of samples. 

			Arguments: 

				y_true:			Dummy variable. It is required by Keras API, but the likelihood 
								computations does not use it.
				y_pred:			The encoding of the model as a single numpy array, *NOT* (X, Zs). 

			Returns:		
				Log latent density of the encoded data in bits per dimension averaged over samples. 

			Comments: 

				The average over the mini-batch is done to balance this term with the log_det.
				Alternatively, one could have scaled log_det by the size of the mini-batch. 
			
		"""

		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), 		tf.float32)
		norm		= d * np.log(2.) 

		# Divide by /d to get per dimension. 
		# Divide by log(2) to go from log base E (default in tensorflow) to log base 2. 
		# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math/log
		normal 		= self.latent.log_density(y_pred) / norm 
		batch_size 	= tf.cast(tf.shape(y_pred)[0], 	tf.float32)
		normal		= normal / batch_size
		return 		- normal


	def loss_log_var_dequant(self, y_true, y_pred): 
		"""
			Variational Dequantization introduces an additional term to the loss function, 
			see equation denominator in (12) from [1]. If the model has a variational 
			dequantization layer, this function computes the dequantization loss, if not,
			this function returns 0. 
			
			Arguments: 

				y_true: 	Dummy variables, required by Keras but not used. 
				y_pred: 	Dummy variables, required by Keras but not used. 

			Returns:		
				Variational Dequantization part of loss function if used (returns 0 otherwise). 

			[1] Flow++: Improving Flow-Based Generative Models with 			https://arxiv.org/pdf/1902.00275
				Variational Dequantization and Architecture Design
			
		"""
		
		vardeqloss = 0
		for layer in self.layers: 
			if isinstance(layer, invtf.dequantize.VariationalDequantize): 
				vardeqloss = layer.loss()
				break

		d			= tf.cast(tf.reduce_prod(y_pred.shape[1:]), 		tf.float32)
		norm		= d * np.log(2.) 
		vardeqloss 	= vardeqloss / norm
		return - vardeqloss




	def log_det(self):	
		"""
			Computes the log determinant by calling all layers. This is called by
			'los_log_det' which uses this to compute normalized negative 
			log determinant.
		"""
		logdet = 0.

		for layer in self.layers: 
			if isinstance(layer, tf.keras.layers.InputLayer): 	continue 
			logdet += layer.log_det()
			
		return logdet





	def add(self, layer): 
		"""
			Adds a layer following all previous layers. 

			Arguments:
				layer:	A layer that supports call, call_inv and log_det.  

			Example: 
					
				> import invtf
				> import tensorflow.keras as keras

				> # Model
				> g = invtf.Generator()
				> g.add(invtf.dequantize.UniformDequantize(input_shape=input_shape)) 
				> g.add(invtf.layers.Normalize()) 
				> g.add(invtf.layers.Conv3DCirc()) 


			Comments: 

				This code is essentially that of keras.Sequential [1]. As explained 
				above the class inherits from Model instead of Sequential to allow
				Multi-Scale architecture with multiple outputs. To simultaneously support
				the Sequential API we have a modified version of the Sequential 'add(..)'
				function. 
		
				The main modification is that the function below allow multiple outputs 
				of MultiScale layers which keras.Sequential does not. 


			TODO:
				(1) Make InvLayer class everyone inherits from with O(1) mem backprop. 

		"""

		if len(self._layers) == 0:
			if not hasattr(layer, "_batch_input_shape"): 
				raise Exception("The first layer should include input dimension, e.g. UniformDequantize(input_shape=X.shape[1:]). ")

		if isinstance(layer, 		keras.layers.InputLayer): 
			raise Exception("Don't add an InputLayer, this is the responsibility of the Generator class. ")

		if not isinstance(layer, 	keras.layers.Layer): 
			raise TypeError("The added layer must be an instance of class Layer. Found: " + str(layer))

		self.built = False
		set_inputs = False

		from tensorflow.python.keras.engine import input_layer
		from tensorflow.python.keras.engine import training_utils
		from tensorflow.python.keras.utils import layer_utils
		from tensorflow.python.util import nest
		from tensorflow.python.util import tf_inspect

		# If list is empty. 
		if not self._layers:	 
			batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
			if batch_shape:
					# Instantiate an input layer.
					x = input_layer.Input(
							batch_shape=batch_shape, dtype=dtype, name=layer.name + '_input')
					# This will build the current layer
					# and create the node connecting the current layer
					# to the input layer we just created.
					layer(x)
					set_inputs = True

			if set_inputs:
				# If an input layer (placeholder) is available.
				if len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) != 1:
					raise ValueError('All layers of Invertible Generator (besides MultiScale) '
													 'should have a single output tensor. ' )
				self.outputs = [
						nest.flatten(layer._inbound_nodes[-1].output_tensors)[0]
				]
				self.inputs = layer_utils.get_source_inputs(self.outputs[0])

		elif self.outputs:
			"""# If the model is being built continuously on top of an input layer:
			# refresh its output.
			output_tensor = layer(self.outputs[0])
	
			# MAIN NEW LINE
			if isinstance(layer, invtf.layers.MultiScale): output_tensor = output_tensor[0]

			if len(nest.flatten(output_tensor)) != 1:
				raise TypeError('All layers of Invertible Gnerator (Besides MultiScale) '
												'should have a single output tensor. ')
			self.outputs = [output_tensor]"""

			# If the model is being built continuously on top of an input layer:
			# refresh its output.
			output_tensor = layer(self.outputs[0])

			# MAIN MODIFICATION.
			Zs = []
			if isinstance(layer, invtf.layers.MultiScale): 
				Zs = [output_tensor[1]]
				output_tensor = output_tensor[0]

			if len(nest.flatten(output_tensor)) != 1:
				raise TypeError('All layers of Invertible Generator (Besides MultiScale) '
												'should have a single output tensor. ')
			self.outputs = [output_tensor] + self.outputs[1:] + Zs


		if self.outputs:
			# True if set_inputs or self._is_graph_network or if adding a layer
			# to an already built deferred seq model.
			self.built = True

		if set_inputs or self._is_graph_network:
			self._init_graph_network(self.inputs, self.outputs, name=self.name)
		else:
			self._layers.append(layer)
		if self._layers:
			self._track_layers(self._layers)

		self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)


	def sample(self, n=1000, fix_latent=True, std=1.0):	
		"""
			Generates fake samples. 

			Arguments: 
				n:				Number of samples (integer)
				fix_latent: 	Generate samples for fixed randomness. If done during  training
								this will allow one to see how a single sample changes during 
								training. 
				std:			The standard deviation of the latent space distribution. Many
								state-of-the-art models [1,2] found sampling at lower variance allows
								yield better looking pictures (but less varied). 
			Returns:
				A number 'n' fake generated samples as a NumPy array. 

			Comments:
				
			

			[1] Glow: Generative Flow with Invertible 1x1 Convolutions				https://arxiv.org/abs/1807.03039
			[2] Large Scale GAN Training for High Fidelity Natural Image Synthesis 	https://arxiv.org/abs/1809.11096
		"""
		output_shape 	= self.layers[-1].output_shape[1:]

		X = self.latent.sample(shape=(n, ) + output_shape, std=std)

		for layer in self.layers[::-1]: 
			if isinstance(layer, keras.layers.InputLayer): continue

			if isinstance(layer, invtf.layers.MultiScale): 
				Z = self.latent.sample(shape=X.shape)
				X = layer.call_inv(X, Z)
			else: 
				X = layer.call_inv(X)

		return np.array(X) 

	def interpolate(self, a, b): 	raise NotImplementedError() 


	def init(self, X):	 # TODO: consider changing this to build and call super(..).build(..) in end?
		"""
			Initializes ActNorm data dependently so output have zero mean and unit 
			variance (channel-wise). Initializes Normalization so activations  have 
			zero mean and unit variance. 

			Arguments:
				X:		Input data, typically a NumPy array. Recommend using around
						a thousand (1000) examples. 

			Comments:

				(1) The code assumes that Normalization happens only once, and it happens
				before any ActNorm layers. 
			
				(2) It would be nice to experimentally investigate the importance of 
				how much data is used to data dependently initialize ActNorm. I believe
				the original Glow code used 250 examples or so. It would be interesting
				to see how big an effect the number of examples has on generative performance.

				(3) For ease of implementation, the ActNorm layers are initialized with
				NumPy on the CPU. Amongst other things, this allow easily writing assertions
				that check the normalization. Implementing this on GPU will speed things,
				however, it seems that this initialization step takes milliseconds for
				rather even big models, so time consumption seems to be negligible. 

		"""


		# Initialize Normalization. 
		for i, layer in enumerate(self.layers): 
			if isinstance(layer, invtf.layers.Normalize): break
			X = layer.call(X)
		
		max = np.max(X.numpy()) 
		layer.scale = 1 / (max / 2)
		X = layer.call(X).numpy()

		assert np.allclose(X.min(), 0, X.max(), 1)

		# Initialize ActNorm layers (done on CPU). 
		for layer in self.layers[i:]:

			if isinstance(layer, invtf.layers.ActNorm):	# do normalization
				if isinstance(X, tf.Tensor): X = X.numpy()
				# normalize channel-wise to mean=0 and var=1. 
				n, h, w, c = X.shape
				for i in range(c):	# go through each channel. 
					mean = np.mean(X[:, :, :, i])
					X[:, :, :, i] = X[:, :, :, i] - mean

					std	= np.std(X[:, :, :, i])
					X[:, :, :, i] = X[:, :, :, i] / std


					std	= np.std(X[:, :, :, i])
					mean = np.mean(X[:, :, :, i])
					assert np.allclose(std, 1) and np.allclose(mean, 0, atol=10**(-2)), (mean, std)

			X = layer.call(X)

			if isinstance(layer, invtf.layers.MultiScale): X = X[0]

	def fit(self, X, **kwargs): 
		"""
			Trains the model to maximize the likelihood of the data. 
			
			Arguments:
				X:	Input data, typically in NumPy format. 

			Comments:
				We are doing unsupervised learning, but the Keras API requires 
				the fit function of Model [1] to take both examples X and labels y. 
				To circumvent this we have pass data as labels. 

				TODO: Investigate if this work-around hurt performance. I believe 
				this just provides the fit function with another pointer to X, which
				would have a negligible effect on performance.

		"""
		return super(Generator, self).fit(X, y=X, **kwargs)	



	"""

		Functions above this point are implement core functionality of training. 
		Functions below this point are convenience functions. 

	"""

	def rec(self, X): 
		"""
			Convenience function. Reconstructs input by first encoding and 
			then decoding it. This is useful for keeping track of numerical 
			instability. 
		
			Arguments:
				X:		Input to be reconstructed, typically NumPy array. 
			
			Returns: 
				Reconstructed input. Theoretically it should be true that 
					np.allclose(X, model.rec(X))
				Unfortunately, numerical issues typically make this test fail,
				however, a similar test with lower numerical precision usually 
				holds: 
					np.allclose(X, model.rec(X), atol=10**(-5), rtol=10**(-2))
				See [1] for details. 


			[1] NumPY.allclose					https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
				
			TODO: 
				Make an argument for predict that specifies if predictions are with
				float32 or float64. Experimentally investigate the size of numerical
				error when using different precision. 

		"""
		X, Zs 	= self.predict(X, dequantize=False) 
		rec 	= self.predict_inv(X, Zs)
		return rec

	def inspect_log_det(self):	
		"""
			Convenience function. Computes how large the contribution of each layer
			is to the total log determinant. This is useful for debugging the 
			log determinant computations of each layer. For example, if one layer
			type accounts for 99% of the entire log-determinant something is 
			probably wrong. 

		"""
		logdet = 0.

		self.ratio = {}
		self.ratio_ = {}

		norm = 32*32*3 * np.log(2)

		for layer in self.layers: 
			val 		= - layer.log_det() / norm
			logdet 		+= val

			key 		= str(type(layer))
			if not key in self.ratio.keys(): self.ratio[key] = 0
			self.ratio[key] += val

		for key in self.ratio.keys():
			self.ratio_[key] = self.ratio[key] / logdet

		for key in self.ratio.keys():
			print(self.ratio_[key], "\t", self.ratio[key], "\t", key)


	def encode(self, X):
		"""
			Convenience function, similar to predict, but output is a single numpy array
			with the same size as original input. 

			Arguments: 

				X:				X Input samples, typically Numpy array.  

			Returns:		
				Numpy array X containing encoding. 

		"""
		X, Zs = self.predict(X)
		X = X.numpy()

		if Zs == []: return X

		Zs = [Z.numpy() for Z in Zs]

		X = X.reshape((X.shape[0], -1))
		Zs = [Z.reshape((Z.shape[0], -1)) for Z in Zs]

		out = np.concatenate([X] + Zs, axis=-1)
		return out 

		


	def check(self, X): 
		"""
			Convenience function. Computes fake, encoding, reconstruction and 
			plots this next to the real data. This can be used to e.g. check
			identity initialization, in this case the encoding and the real 
			image should look alike (also true for reconstruction). 
			
			Comments:
				The images are drawn assuming X is in [0, 255] and that 
				the normalization divides by 127.5 and substracts 1/2. 

		"""
	
		fig, ax = plt.subplots(1, 4)
		for i in range(4): ax[i].axis("off")
		img_shape = X.shape[1:]
		#fig.canvas.manager.window.wm_geometry("+2500+0")

		fake = self.sample(1, fix_latent=True)

		ax[0].imshow(fake.reshape(img_shape)/255)
		ax[0].set_title("Fake")

		ax[1].imshow(X[0].reshape(img_shape)/255)
		ax[1].set_title("Real")

		ax[2].imshow(self.rec(X[:1]).reshape(img_shape)/255)
		ax[2].set_title("Reconstruction")

		ax[3].imshow(self.encode(X[:1]).reshape(img_shape)/2+1/2) 
		ax[3].set_title("Encoding")

		plt.tight_layout()
		plt.show()


