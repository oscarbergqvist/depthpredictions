import tensorflow as tf
import numpy as np
import scipy.io as sio


with tf.variable_scope("read_data"): 

	def load_data():
		''' 
		Loads data from .mat file. For debugging dataset, substitute file
		and mat_contents for 'subset_depthdata.mat' and 'sub_' respectively.

		Full data: 
		- 'depth_datasetNYUv2.mat', 'images', 'depths'

		Returns: 
			- images [no_images = 1449,height = 480, width = 640, channels = 3]
			- depth_maps [no_images = 1449,height = 480, width = 640]

		Comments: 
		'''
		mat_contents = sio.loadmat('depth_datasetNYUv2.mat')

		images = mat_contents['images']
		depth_maps = mat_contents['depths']

		images = np.moveaxis(images, -1, 0)
		depth_maps = np.moveaxis(depth_maps, -1, 0)

		#images = np.moveaxis(images, 1, 2)
		#depth_maps = np.moveaxis(depth_maps, 1, 2)
		# scipy.misc.imresize()

		assert images.shape[0] == depth_maps.shape[0]

		
		return images, depth_maps

	def split_data(images, labels):
		'''
		Splits data in three partitions: training_(images+labels), 
		test_(images+labels), validation_(images+labels)

		Returns: 
			- training_images
			- training_labels
			- test_images
			- test_labels
			- validation_images
			- validation_labels
		'''
		no_images = images.shape[0]
		val_or_test_partition = (no_images // 5)
		end_of_training_partition = no_images - 2* val_or_test_partition

		a = end_of_training_partition
		b = end_of_training_partition + val_or_test_partition

		[training_images, test_images, validation_images] = np.split(images, [a, b], axis = 0)

		[training_labels, test_labels, validation_labels] = np.split(labels, [a, b], axis = 0)

		return training_images, training_labels, test_images, test_labels, validation_images, validation_labels


	def generate_dataset(images, labels):
		'''
		Generates a dataset from placeholders, resizes pictures and returns 
		dataset and placeholders. 
		''' 
		images_placeholder = tf.placeholder(images.dtype, shape = [None, 480, 640, 3])
		labels_placeholder = tf.placeholder(labels.dtype, shape = [None, 480, 640])
		batch_size_placeholder = tf.placeholder(tf.int64, ())
		dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder)).batch(batch_size_placeholder).repeat()

		dataset = dataset.map(_resizing_function)

		return dataset, images_placeholder, labels_placeholder, batch_size_placeholder

	def _resizing_function(image,label):
		label = label[..., np.newaxis]
		resized_image = tf.image.resize_images(image, [223, 303])
		resized_label = tf.image.resize_images(label, [55, 74])


		return resized_image,resized_label

	# Load all data
	rgb_images, depth_maps = load_data()
	training_images, training_labels, test_images, test_labels, validation_images, validation_labels = split_data(rgb_images, depth_maps)

	# Ugly-getters
	def get_data(images = training_images, labels = training_labels):
		return images, labels

	def get_validation_data(images = validation_images, labels = validation_labels):
		return images, labels

	def get_test_data(images = test_images, labels = test_labels):
		return images, labels



	'''Approach: one dataset, feed different arrays:
	''' 
	dataset, images_placeholder, labels_placeholder, batch_size_placeholder = generate_dataset(training_images, training_labels)
	iterator = dataset.make_initializable_iterator()
	(image_batch, depthmap_batch) = iterator.get_next()
	batch_size = 4



	'''Approach: three datasets, re-initializable iterator 
	'''
	'''Create datasets and placeholders '''
	'''training_dataset = generate_dataset(training_images, training_labels)
	test_dataset = generate_dataset(test_images, test_labels)
	validation_dataset = generate_dataset(validation_images, validation_labels)
	'''



	''' Debuggers 
	call_shape = print(iterator.output_shapes)

	feed = {images_placeholder : training_images, 
			labels_placeholder : training_labels,
			batch_size_placeholder : batch_size}


	with tf.Session() as sess:
		sess.run(iterator.initializer, feed_dict = feed)
		sess.run(iterator.get_next())

	'''














