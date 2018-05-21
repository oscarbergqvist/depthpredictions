import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle

print(tf.VERSION)

# Returns all available images and their corresponding depths
# In total 1449 pictures
def read_data():
	mat_contents = sio.loadmat('depth_datasetNYUv2.mat')

	images = mat_contents['images']
	depths = mat_contents['depths']

	images = np.moveaxis(images, -1, 0)
	depths = np.moveaxis(depths, -1, 0)

	assert images.shape[0] == depths.shape[0]

	images_placeholder = tf.placeholder(images.dtype, images.shape)
	depths_placeholder = tf.placeholder(depths.dtype, depths.shape)

	dataset = tf.data.Dataset.from_tensor_slices((images_placeholder,
												 depths_placeholder))
	dataset = dataset.map(_resizing_function)

	iterate_data = dataset.make_initializable_iterator()

	return iterate_data, images, depths, images_placeholder, depths_placeholder


# Returns 15 images and their corresponding depths for debugging purposes
def read_debug_data():
	mat_contents = sio.loadmat('subset_depthdata.mat')

	images = mat_contents['sub_images']
	depths = mat_contents['sub_depths']

	images = np.moveaxis(images, -1, 0)
	depths = np.moveaxis(depths, -1, 0)

	assert images.shape[0] == depths.shape[0]

	images_placeholder = tf.placeholder(images.dtype, images.shape)
	depths_placeholder = tf.placeholder(depths.dtype, depths.shape)

	dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, 
													depths_placeholder))

	new_height = tf.constant(value = new_height, dtype = tf.float32)
	new_width = tf.constant(value = new_width, dtype= tf.float32)

	dataset = dataset.map(_resizing_function)

	
	#iterate_data = sub_dataset.make_initializable_iterator()


	return dataset, images, depths, images_placeholder, depths_placeholder

def _resizing_function(image,label, new_height = 303, new_width = 227):
	label = label[..., np.newaxis]
	resized_image = tf.image.resize_images(image, [new_height, new_width])
	resized_label = tf.image.resize_images(label, [new_height, new_width])


	return resized_image,resized_label


(dataset, images, depths, images_placeholder, depths_placeholder) = read_debug_data()	


print(images.shape)
print(depths.shape)


iterate_data = dataset.make_initializable_iterator()
next_element = iterate_data.get_next()
call_shape = print(iterate_data.output_shapes)


with tf.Session() as sess:
	sess.run(iterate_data.initializer, feed_dict={images_placeholder:images, 
													depths_placeholder:depths})
	sess.run(call_shape)

