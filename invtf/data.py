import tensorflow as tf
import os

def load_image_dataset(folder, epochs=1,new_image_size=(64, 64),batch_size=32,shuffle=True,):
	def _parse_function(filename):
		image_string = tf.io.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string)
		image_resized = tf.image.resize(image_decoded, new_size)
		return image_resized

	files = ['{}/{}'.format(folder,f) for f in os.listdir(
		folder) if os.path.isfile(os.path.join(folder, f))]
	dataset = tf.data.Dataset.from_tensor_slices(tf.constant(files))
	if shuffle == True:
		dataset = dataset.shuffle(buffer_size=100)
	dataset = dataset.repeat(count=epochs)
	dataset = dataset.map(map_func=_parse_function,num_parallel_calls=4)
	dataset = dataset.prefetch(buffer_size=batch_size)
	dataset = dataset.batch(batch_size=batch_size)
	return iter(dataset),(len(files)//batch_size)
