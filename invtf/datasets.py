import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets 
import os 
import tensorflow as tf 


"""
	TODO: make comments on keras datasets (mnist, fmnist, cifar10+100) and
	those from the glow github repository (imagenet32, celeb). 

"""

def mnist(digit=-1):  # -1 => use all classes. 

	(X, y), (X_test, y_test) = datasets.mnist.load_data()
	X = X.astype(np.float32).reshape(60000, 28, 28, 1)

	if digit > -1: 	return X[y 	 == digit]
	else:			return X

def fmnist(digit=-1): 

	(X, y), (X_test, y_test) = datasets.fashion_mnist.load_data()
	X = X.astype(np.float32).reshape(60000, 28, 28, 1)

	if digit > -1: 	return X	 [y 	 == digit]
	else: 			return X
	
def cifar10(): 
	(X, _), (X_test, _) = datasets.cifar10.load_data()
	return X.astype(np.float32)

def cifar100(): 
	(X, y), (X_test, y_test) = datasets.cifar100.load_data()
	return X.astype(np.float32)


"""
	rewrite these to be functions. 

"""

# Helper to load tf records. 
# FROM https://github.com/openai/glow/blob/master/data_loaders/get_data.py#L10
# they prepared the records. 
def parse_tfrecord_tf(record, res, rnd_crop):
	features = tf.io.parse_single_example(record, features={
		'shape': tf.io.FixedLenFeature([3], tf.int64),
		'data': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([1], tf.int64)})
	# label is always 0 if uncondtional
	# to get CelebA attr, add 'attr': tf.FixedLenFeature([40], tf.int64)
	data, label, shape = features['data'], features['label'], features['shape']
	label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
	img = tf.io.decode_raw(data, tf.uint8)
	if rnd_crop:
		# For LSUN Realnvp only - random crop
		img = tf.reshape(img, shape)
		img = tf.random_crop(img, [res, res, 3])
		img = tf.reshape(img, [res, res, 3])
	return img, label  # to get CelebA attr, also return attr



def celeba(n=1000, resolution=256):  # for now only downloads Glow version. 

	if not resolution in [16, 32, 64, 128, 256]: raise NotImplementedError("Only supports resolution 16, 32, 64, 128 and 256. ")

	from tensorflow.python.keras.utils.data_utils import get_file
	import tarfile

	print("Starting to download and extract celeb256, it is 4GB, so this might take a long time. ")
	print("Go grab a cup of coffee. ")
	loc = get_file("celeb256.tar", "https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar") # located in ~/.keras/celeb256.tar
	tarf = tarfile.open(loc)
	loc = loc.replace("celeb256.tar", "")
	print("Extracting tar into %s"%loc)
	#tarf.extractall(loc) # make check so if it is already extracted don't do it again. 
	print("Done extracting into %s"%loc)


	folder = "/home/maximin/.keras/datasets/celeba-tfr/train/"
	files = [folder + f for f in os.listdir(folder)]
	
	ds = tf.data.TFRecordDataset( files )

	data = np.zeros((n, 256,256,3), dtype=np.uint8)
	for i, raw_record in enumerate(ds.take(n)): # refactor for loop away, slow. 
		img, lbl = parse_tfrecord_tf(raw_record, 256, False)
		img = tf.reshape(img, ((256,256,3)))
		data[i] =  img

	data = data.astype(np.float32)

	# initial resolution is 256x256, downsample if other resolution is specified. [:, ::8, ::8, :]
	if resolution == 256: return data
	if resolution == 128: return data[:, ::2, ::2, :]
	if resolution == 64: return data[:, ::4, ::4, :]
	if resolution == 32: return data[:, ::8, ::8, :]
	if resolution == 16: return data[:, ::16, ::16, :]
	else: raise Exception("Did not understand resolution %s"%str(resolution))

	

def generator(self): raise NotImplementedError()


class imagenet():  # get from ?? 

	def __init__(self, bits=8, resolution=64): 
		if resolution not in {32, 64}: raise NotImplementedError("Only support resolution 32 or 64, not %i. "%resolution)

		if not os.path.exists("datasets/"): os.makedirs("datasets/")

		from tensorflow.python.keras.utils.data_utils import get_file
		import tarfile

		if resolution == 32:  # 4 GB
			print("Starting to download and extract imagenet32, it is 4GB, so this might take a long time. ")
			print("Go grab a cup of coffee. ")
			loc = get_file("imagenet32train.tar", "http://image-net.org/small/train_32x32.tar") # save this location for later use? 
			tarf = tarfile.open(loc)
			tarf.extract_all()

		if resolution == 64:  # 12 GB (might take a very long time)
			print("Starting to download and extract imagenet64, it is 12GB, so this might take a very long time. ")
			print("Go hit the gym or go for a walk. ")
			loc = get_file("imagenet32train.tar", "http://image-net.org/small/train_64x64.tar") # save this location for later use? 
			tarf = tarfile.open(loc)
			tarf.extract_all()
	
	
	def images(self, n=10000):  # how many to load 

		pass


	def generator(self): raise NotImplementedError() # implement a generator later. 
	






############### TOY DATASET #################

# Easy
class Normal(): 
	def __init__(self): pass 
	def sample(self, n=1000, mean=0, std=1): return np.random.normal(mean, std, size=(n, 2)).astype(np.float32)

class Uniform(): 
	def __init__(self): pass 
	def sample(self, n=1000, a=0, b=1): 	return np.random.uniform(a, b, size=(n, 2)).astype(np.float32)


# Harder 
"""
	Generate several Gaussians located uniformly on 2d unit circle. 
	See FFJORD page 5 for an example. 
	For simplicity the one below uses 4 Gaussians. 
	TODO: Refactor and generalize to 'n' Gaussians. 

"""
class Gaussians():  
	
	def sample(self, n=100, std=.2): 

		def gaussian(): return np.random.normal(0, std, size=(n, 2))

		# generate gaussians uniformly distributed on circle. 
		# make 'four gaussians' for simplicity. 
		e1 = np.array([0, 1]).reshape(1, 2)
		e2 = np.array([1, 0]).reshape(1, 2)
		
		g1 = gaussian() + e1 
		g2 = gaussian() - e1 
		g3 = gaussian() + e2
		g4 = gaussian() - e2

		return np.concatenate((g1, g2, g3, g4), axis=0).astype(np.float32)

"""
	Generate checkerboard so alternating cells having points. 
	See FFJORD page 5 for an example. 
	For simplicity assumes 4x4 gird. 
	TODO: Refactor and generalize to 'n' cells. 
"""
class Checkboard(): 	 

	def __init__(self): pass 
	def sample(self, n=100): 
		def uniform(): return np.random.uniform(0, 1, size=(n, 2))

		e1 = np.array([0, 1]).reshape(1, 2)
		e2 = np.array([1, 0]).reshape(1, 2)

		u1 = uniform() - 2*e1 
		u2 = uniform() - 2*e2
		u3 = uniform() - 2*e1 - 2*e2
		u4 = uniform() - e1 - e2
		u5 = uniform() + e1 - e2
		u6 = uniform() - e1 + e2
		u7 = uniform() + e1 + e2
		u8 = uniform() 

		return np.concatenate((u1, u2, u3, u4, u5, u6, u7, u8), axis=0).astype(np.float32)


"""
	Generates to "reflected" spirals, see FFJORD page 5 for example. 
	Note: 	
		the reflected spirals are produced using different samples. 
		density is not uniform throughout 'the spiral', i.e., there are more points closer to center. 

"""
class TwoSpirals():  	

	def sample(self): 

		offset = np.array([0, 0.4])
	
		n = np.random.uniform(0, 1, size=(1000, 1)) *4*np.pi# uniform line
		x = -np.cos(n) * n
		y = np.sin(n) * n
		spiral1 =  np.concatenate((x,y), axis=1) + offset

		n = np.random.uniform(0, 1, size=(1000, 1)) *4*np.pi# uniform line
		x = np.cos(n) * n
		y = -np.sin(n) * n
		spiral2 =  np.concatenate((x,y), axis=1) - offset

		return np.concatenate((spiral1, spiral2), axis=0).astype(np.float32)




if __name__ == "__main__": 


	X  = celeba.load(10)
	print(X.shape)

	plt.imshow(X[3])
	plt.show()

	#imagenet(bits=8, resolution=32)
	
