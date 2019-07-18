import tensorflow as tf
import tensorflow.keras as keras 
import numpy as np

class Latent(): 
	"""
		Invertible Generative Models assumes some latent distribution. 
		This is typically Gaussian, but in principle, any distribution that 
		support sampling and log density computations can be used. 
	
		For example, one could use the Logistic distribution as done in [1]. 
		It is also possible to use another flow model as latent distribution. 

		[1] https://arxiv.org/abs/1410.8516
	"""
	def log_density(self, X): 								raise NotImplementedError()
	def sample(self, shape, fix_latent=False, std=1.0): 	raise NotImplementedError()


class Normal(Latent):  
	"""
		Normal Distribution with zero mean and unit std by default. 
	"""

	def __init__(self, std=1): 
		self.std 	= tf.dtypes.cast(std, dtype=tf.float32)
		self.latent = None

	def log_density(self, X): 
		return tf.math.reduce_sum( -1/2 * (X**2/self.std**2 + tf.dtypes.cast(tf.math.log(2*np.pi*self.std**2), dtype=tf.float32)) )

	def sample(self, shape, fix_latent=False, std=None): 
		if std is None: std = self.std
		if self.latent is None: 	self.latent = tf.random.normal(shape, 0, std)
		elif not fix_latent: 		self.latent = tf.random.normal(shape, 0, std)

		return self.latent


class Logistic(Latent): 
	"""
		Logistic Distribution with zero mean and unit std as used in [1]. 

		[1] https://arxiv.org/abs/1410.8516

		Implementation comments: 
			
			Logistic distribution is not in tensorflow 2.0.0 beta so for now it 
			is computed using numpy. We could also compute by transforming normal.
			This would allow GPU only sampling. 
	"""

	def __init__(self):  
		self.latent = None

	def log_density(self, X):
		return tf.math.reduce_sum( - tf.math.log(1 + tf.exp(X)) - tf.math.log(1 + tf.exp(-X)) )

	def sample(self, shape, fix_latent=False, std=1.0):	 
		if self.latent is None: 	self.latent = np.random.logistic(0, std, shape).astype(np.float32)
		elif not fix_latent: 		self.latent = np.random.logistic(0, std, shape).astype(np.float32)

		return self.latent

	
