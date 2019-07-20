import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np
from tensorflow.keras.layers import ReLU
from invtf.override import print_summary
from invtf.coupling_strategy import *


#TODO Write unit tests
class InvLayer(keras.layers.Layer):     # changed name 
	"""
		This is a virtual class from which all layer classes need to inherit
		It has the function `compute gradients` which is used for constant 
		memory backprop.
	"""
	# This is default behaviour in Python. 
	def __init__(self,**kwargs): super(InvLayer, self).__init__(**kwargs)

	def call(self,X): 		raise NotImplementedError()
	def call_inv(self,X): 	raise NotImplementedError()
	def log_det(self): 		raise NotImplementedError()

	def compute_gradients(self,x,dy,regularizer=None,scaling=1):  
		'''
		Computes gradients for backward pass
		Args:
			x - tensor compatible with forward pass, input to the layer
			dy - incoming gradient from backprop
			regularizer - function, indicates dependence of loss on weights of layer
		Returns
			dy - gradients wrt input, to be backpropagated
			grads - gradients wrt weights
		'''
		#TODO check if log_det of AffineCouplingLayer depends needs a regularizer.
		with tf.GradientTape() as tape:
			tape.watch(x)
			y_ = self.call(x)   #Required to register the operation onto the gradient tape
		grads_combined = tape.gradient(y_,[x]+self.trainable_variables,output_gradients=dy)
		dy,grads = grads_combined[0],grads_combined[1:]

		if regularizer is not None:
			with tf.GradientTape() as tape:
				reg = -regularizer()/scaling
			#print(self.trainable_variables)
			grads_wrt_reg = tape.gradient(reg, self.trainable_variables)
			grads = [a[0]+a[1] for a in zip(grads,grads_wrt_reg) if a[1] is not None]
		return dy,grads


class ReduceNumBits(InvLayer): 
	"""
		Glow used 5 bit variant of CelebA. 
		Flow++ had 3 and 5 bit variants of ImageNet. 
		These lower bit variants allow better dimensionality reduction. 
		This layer should be the first within the model. 

		This also means subsequent normalization needs to divide by less. 
		In this sense likelihood is incomparable between different number of bits. 

		It seems to work, but it is a bit instable. 
	"""
	def __init__(self, bits=5, **kwargs):  # assumes input has 8 bits. 
		self.bits = 5
		super(ReduceNumBits, self).__init__(**kwargs)

	def call(self, X): 
		X = tf.dtypes.cast(X, dtype=np.float32)
		return X // ( 2**(8-self.bits) )

	def call_inv(self, Z): 
		# THIS PART IS NOT INVERTIBLE!!
		return Z * (2**(8-self.bits))

	def log_det(self): return 0. 
		


class ActNorm(InvLayer): 

	"""
		The exp parameter allows the scaling to be exp(s) \odot X. 
		This cancels out the log in the log_det computations. 
	"""

	def __init__(self, exp=False, **kwargs): 
		self.exp = exp
		super(ActNorm, self).__init__(**kwargs)

	def build(self, input_shape): 

		n, h, w, c = input_shape
		self.h = h
		self.w = w

		self.s = self.add_weight(shape=c, 	initializer='ones', name="affine_scale") 
		self.b = self.add_weight(shape=c, 	initializer='zero', name="affine_bias")

		super(ActNorm, self).build(input_shape)
		self.built = True

	def call(self, X): 		return X * self.s + self.b
	def call_inv(self, Z):  return (Z - self.b) / self.s

	def log_det(self): 		return self.h * self.w * tf.reduce_sum(tf.math.log(tf.abs(self.s)))

	def compute_output_shape(self, input_shape): return input_shape






"""
	Known issue with multi-scale architecture. 
	The log-det computations normalizes wrt full dimension. 

"""

class Linear(InvLayer): 

	def __init__(self, **kwargs): super(Linear, self).__init__(**kwargs)

	def build(self, input_shape): 

		assert len(input_shape) == 2
		_, d = input_shape

		self.W = self.add_weight(shape=(d, d), 	initializer='identity', name="linear_weight")
		self.b = self.add_weight(shape=(d), 	initializer='zero',		name="linear_bias")
		
		super(Linear, self).build(input_shape)
		self.built = True

	def call(self, X): 		return X @ self.W + self.b 

	def call_inv(self, Z):  return (Z - self.b) @ tf.linalg.inv(self.W)

	def jacobian(self):		return self.W

	def log_det(self): 		return tf.math.log(tf.abs(tf.linalg.det(self.jacobian())))

	def compute_output_shape(self, input_shape): 
		self.output_shape = input_shape
		return input_shape


class Affine(InvLayer): 

	"""
		The exp parameter allows the scaling to be exp(s) \odot X. 
		This cancels out the log in the log_det computations. 
	"""

	def __init__(self, exp=False, **kwargs): 
		self.exp = exp
		super(Affine, self).__init__(**kwargs)

	def build(self, input_shape): 

		#assert len(input_shape) == 2
		d = input_shape[1:]

		self.w = self.add_weight(shape=d, 	initializer='ones', name="affine_scale") 
		self.b = self.add_weight(shape=d, 	initializer='zero', name="affine_bias")

		super(Affine, self).build(input_shape)
		self.built = True

	def call(self, X): 		
		if self.exp: 	return X * tf.exp(self.w) + self.b 
		else: 			return X * self.w 		  + self.b

	def call_inv(self, Z):  
		if self.exp:	return (Z - self.b) / tf.exp(self.w)
		else: 			return (Z - self.b) / self.w

	def jacobian(self):		return self.w

	def eigenvalues(self): 	return self.w

	def log_det(self): 		
		if self.exp: 	return tf.reduce_sum(tf.abs(self.eigenvalues()))
		else: 			return tf.reduce_sum(tf.math.log(tf.abs(self.eigenvalues())))

	def compute_output_shape(self, input_shape): 
		self.output_shape = input_shape
		return input_shape



class Inv1x1Conv(InvLayer):  
	"""
		Based on Glow page 11 appendix B. 
		It is possible to speed up determinant computation by using PLU or QR decomposition
		as proposed in Glow and Emerging Conv papers respectively. 

		Add bias to this operation? Try to see if it makes any difference. 

		Try to compare speed / numerical stability etc for different implementations: 

			1. PLU decomposition
			2. QR
			3. Normal determinant O(c^3)
			4. tensordot vs conv2d. 
			5. Identity Initialization. 
	"""

	def build(self, input_shape): 
		_, h, w, c = input_shape
		self.c = c
		self.h = h
		self.w = w

		# random orthogonal matrix 
		# check if tf.linalg.qr and tf.linalg.lu are more stable than scipy. 
		self.kernel 	= self.add_weight(initializer=keras.initializers.Orthogonal(), shape=(c, c), name="inv_1x1_conv_P")
	
		super(Inv1x1Conv, self).build(input_shape)
		self.built = True

	def call(self, X): 	
		_W = tf.reshape(self.kernel, (1,1, self.c, self.c))
		return tf.nn.conv2d(X, _W, [1,1,1,1], "SAME")

	def call_inv(self, Z):  
		# TODO: only compute inverse when kernel is updated. 
		self.kernel_inv = tf.dtypes.cast(tf.linalg.inv(tf.dtypes.cast(self.kernel, dtype=tf.float64)), dtype=tf.float32) 
		_W = tf.reshape(self.kernel_inv, (1,1, self.c, self.c))
		return tf.nn.conv2d(Z, _W, [1,1,1,1], "SAME")

	def log_det(self): 		  # det computations are way too instable here.. 
		return self.h * self.w * tf.math.log(tf.abs( tf.linalg.det(self.kernel) ))   

	def compute_output_shape(self, input_shape): return input_shape




class Inv1x1ConvPLU(InvLayer):  
	"""
		Based on Glow page 11 appendix B. 
		It is possible to speed up determinant computation by using PLU or QR decomposition
		as proposed in Glow and Emerging Conv papers respectively. 

		Add bias to this operation? Try to see if it makes any difference. 

		Try to compare speed / numerical stability etc for different implementations: 

			1. PLU decomposition
			2. QR
			3. Normal determinant O(c^3)
			4. tensordot vs conv2d. 
	"""

	def build(self, input_shape): 
		_, h, w, c = input_shape
		self.c = c
		self.h = h
		self.w = w

		# random orthogonal matrix 
		# check if tf.linalg.qr and tf.linalg.lu are more stable than scipy. 
		import scipy
		w 		= scipy.linalg.qr(np.random.normal(0, 1, (self.c, self.c)))[0].astype(np.float32)
		P, L, U = scipy.linalg.lu(w)

		def init_P(self, shape=None, dtype=None): return P
		def init_L(self, shape=None, dtype=None): return L
		def init_U(self, shape=None, dtype=None): return U

		# use the PLU decomposition as they do in the article? 
		# I don't think the non PLU cae is stable enough? 

		self.P = self.add_weight(initializer=init_P, shape=P.shape, name="inv_1x1_conv_P", trainable=False)
		self.L = self.add_weight(initializer=init_L, shape=L.shape, name="inv_1x1_conv_L")
		self.U = self.add_weight(initializer=init_U, shape=U.shape, name="inv_1x1_conv_U")

		L_mask = tf.constant(np.triu(np.ones((c,c)), k=+1), dtype=tf.float32)
		P_mask = tf.constant(np.tril(np.ones((c,c)), k=-1), dtype=tf.float32)
		I 	   = tf.constant(np.identity(c), dtype=tf.float32)

		self.P = self.P * P_mask + I
		self.L = self.L * L_mask + I

		self.kernel = self.P @ self.L @ self.U # which order of matrix mult is faster? P is permutation and thus very sparse. 

		self.P_inv = tf.linalg.inv(tf.dtypes.cast(P, dtype=tf.float64))
		self.L_inv = tf.linalg.inv(tf.dtypes.cast(L, dtype=tf.float64))
		self.U_inv = tf.linalg.inv(tf.dtypes.cast(U, dtype=tf.float64))

		self.kernel_inv 	= tf.linalg.inv(self.kernel) # tf.dtypes.cast(self.U_inv @ self.L_inv @ self.P_inv, dtype=tf.float32)

		#self.I_ 			= self.kernel @ tf.linalg.inv(self.kernel)
		#self.I 				= self.kernel @ self.kernel_inv
	
		super(Inv1x1Conv, self).build(input_shape)
		self.built = True

	def call(self, X): 	
		_W = tf.reshape(self.kernel, (1,1, self.c, self.c))
		return tf.nn.conv2d(X, _W, [1,1,1,1], "SAME")

	def call_inv(self, Z):  
		_W = tf.reshape(self.kernel_inv, (1,1, self.c, self.c))
		return tf.nn.conv2d(Z, _W, [1,1,1,1], "SAME")

	def log_det(self): 		  # det computations are way too instable here.. 
		return self.h * self.w * tf.math.log(tf.abs( tf.linalg.det(self.kernel) ))   # Looks fine? 

	def compute_output_shape(self, input_shape): return input_shape




class AdditiveCouplingReLU(InvLayer): 

	unique_id = 1

	def __init__(self, part=0, sign=+1, strategy=SplitChannelsStrategy()): # strategy: alternate / split  ;; alternate does odd/even, split has upper/lower. 
		super(AdditiveCouplingReLU, self).__init__(name="add_coupling_relu_%i"%AdditiveCouplingReLU.unique_id)
		AdditiveCouplingReLU.unique_id += 1
		self.part 		= part 
		self.strategy 	= strategy
		self.relu		= keras.layers.ReLU()
		self.sign		= sign

	def call(self, X): 		
		shape 	= tf.shape(X)

		x0, x1 = self.strategy.split(X)

		if self.part == 0: x0 		= x0 + self.relu(x1) * self.sign
		if self.part == 1: x1 		= x1 + self.relu(x0) * self.sign

		X = self.strategy.combine(x0, x1)

		return X

	def call_inv(self, Z):	 
		z0, z1 = self.strategy.split(Z)
		
		if self.part == 0: z0 		= z0 - self.relu(z1) * self.sign
		if self.part == 1: z1 		= z1 - self.relu(z0) * self.sign

		Z = self.strategy.combine(z0, z1)

		return Z


	def log_det(self): 		return 0. 

	def compute_output_shape(self, input_shape): return input_shape






"""
	Try different techniques: I'm implementing the simplest case, just reshape to desired shape. 
	TODO: Implement the following Squeeze strategies: 
		- RealNVP: original Squeeze, different to what we do below.
		- Downscale images, e.g. alternate pixels and have 4 lower dim images and stack them. 
		- ... 
"""
class Squeeze(InvLayer): 

	def call(self, X): 
		n, self.w, self.h, self.c = X.shape
		return tf.reshape(X, [-1, self.w//2, self.h//2, self.c*4])

	def call_inv(self, X): 
		return tf.reshape(X, [-1, self.w, self.h, self.c])
		
	def log_det(self): return 0. 


class UnSqueeze(InvLayer): 

	def call(self, X): 
		n, self.w, self.h, self.c = X.shape
		return tf.reshape(X, [-1, self.w*2, self.h*2, self.c//4])

	def call_inv(self, X): 
		return tf.reshape(X, [-1, self.w, self.h, self.c])
		
	def log_det(self): return 0. 



class Normalize(InvLayer):  # normalizes data after dequantization. 
	"""

	"""

	def __init__(self, bits=None, target=[-1,+1], scale=127.5, input_shape=[], **kwargs):
		super(Normalize, self).__init__(**kwargs)
		self.target = target

		if bits is None: 	self.scale = 1 / 127.5 
		else: 				self.scale = float(1 / ( 2**(bits -1)))

	def build(self, input_shape): 
		self.d = tf.dtypes.cast(tf.math.reduce_prod(input_shape[1:]), dtype=tf.float32)
		print(self.d)
		super(Normalize, self).build(input_shape=input_shape)
		self.built = True

	def call(self, X):  
		X 			= X * self.scale  - 1
		return X

	def call_inv(self, Z): 
		Z = Z + 1
		Z = Z / self.scale
		return Z

	def log_det(self): 
		return self.d * tf.dtypes.cast(tf.math.log(self.scale), dtype=tf.float32)


class MultiScale(InvLayer): 

	def call(self, X):  # TODO: have different strategies here, and combine it with how coupling layer works? 
		n, w, h, c = X.shape
		Z = X[:, :, :, c//2:]
		X = X[:, :, :, :c//2]
		return X, Z
	
	def call_inv(self, X, Z): 
		return tf.concat((X, Z), axis=-1)

	def compute_output_shape(self, input_shape): 
		n, h, w, c = input_shape
		return (n, h, w, c//2)

	def log_det(self): return 0.




"""
	There's an issue with scaling, which intuitively makes step-size VERY small. 
"""
class Conv3DCirc(InvLayer): 

	def __init__(self,trainable=True, **kwargs): 
		self.built = False
		super(Conv3DCirc, self).__init__(**kwargs)

	def build(self, input_shape): 
		self.scale = np.sqrt(np.prod(input_shape[1:])) 

		def identitiy_initializer_real(shape, dtype=None):
			return (tf.math.real(tf.signal.ifft3d(tf.ones(shape, dtype=tf.complex64)*self.scale))) 

		self.w_real     = self.add_variable(name="w_real",shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True)

		super(Conv3DCirc, self).build(input_shape)
		self.built = True


	def call(self, X): 
		w  = tf.cast(self.w_real, dtype=tf.complex64)
		w  = tf.signal.fft3d(w / self.scale)

		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X / self.scale) 
		X = X * w
		X = tf.signal.ifft3d(X * self.scale ) 
		X = tf.math.real(X)
		return X

	def call_inv(self, X): 
		X = tf.cast(X, dtype=tf.complex64)
		X = tf.signal.fft3d(X * self.scale ) 

		w  = tf.cast(self.w_real, dtype=tf.complex64)
		w  = tf.signal.fft3d(w / self.scale)

		X = X / w

		X = tf.signal.ifft3d(X / self.scale)   
		X = tf.math.real(X)
		return X

	def log_det(self):  return tf.math.reduce_sum( tf.math.log( tf.math.abs( tf.signal.fft3d( tf.cast( self.w_real / self.scale, dtype=tf.complex64)))))    

	def compute_output_shape(self, input_shape): return tf.TensorShape(input_shape[1:])


class ResidualConv3DCirc(Conv3DCirc): 

	# TODO: See if there is any difference in this compared to having
	# a having +1 on the diagonal in Fourier space. What is most numerically
	# stable? 
	def call(self, X): return X + super(ResidualConv3DCirc, self).call(X)  

	# use fixed point iteration algorithm from iResNet
	def call_inv(self, X):  raise NotImplementedError()

	# use the derivations in iResNet to fix this. 
	def log_det(self):   	raise NotImplementedError() # return tf.math.reduce_sum(tf.math.log(tf.math.abs(tf.signal.fft3d(tf.cast(self.w_real/self.scale,dtype=tf.complex64)))))    #Need to return EagerTensor

	def build(self, input_shape): 
		self.scale = np.sqrt(np.prod(input_shape[1:])) # np.sqrt(np.prod([a.value for a in input_shape[1:]]))

		# todo; change to [[[1, 0000],[0000], [000]] 

		def identitiy_initializer_real(shape, dtype=None):
			return (tf.math.real(tf.signal.ifft3d(tf.ones(shape, dtype=tf.complex64)*self.scale))) 

		class SpectralNormalization(keras.constraints.Constraint):
			def __call__(self, w): return w / tf.math.reduce_max(w) # TODO: This needs to be done in fourier space. 

		self.w_real     = self.add_variable(name="w_real",shape=input_shape[1:], initializer=identitiy_initializer_real, trainable=True, constraint=SpectralNormalization())
		# self.w    = tf.cast(self.w_real, dtype=tf.complex64)  #hacky way to initialize real w and actual w, since tf does weird stuff if 'variable' is modified
		# self.w    = tf.signal.fft3d(self.w / self.scale)
		super(Conv3DCirc, self).build(input_shape)
		self.built = True


class Reshape(InvLayer):
	def __init__(self, shape): 
		self.shape = shape
		super(Reshape, self).__init__()

	def call(self, X): 
		self.prev_shape = X.shape
		return tf.reshape(X, (-1, ) + self.shape)

	def log_det(self): return .0

	def call_inv(self, X): return tf.reshape(X, self.input_shape)





class InvResNet(InvLayer): 			


	pass # model should automatically use gradient checkpointing if this is used. 


