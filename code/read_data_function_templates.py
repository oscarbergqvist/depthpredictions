####### READ DATA ########

### PLACEHOLDERS

# batch_size_placeholder: 1d placeholder for batchsize with shape [1] & dtype = tf.int32
# images_placeholder: 3d placeholder for images with shape [None, 223, 303, 3] & dtype = tf.float32 
# depthmaps_placeholder: 2d placeholder for depthmaps with dim [None, 55, 74] & dtype = tf.float32


### DATASET GLOBAL VARIABLES 

dataset = (tf.data.Dataset.from_tensor_slices((images_placeholder, depthmaps_placeholder))
						.batch(batch_size_placeholder)
						.repeat())
iterator = dataset.make_initializable_iterator()

(image_batch, depthmap_batch) = iterator.get_next()   # placeholders with shapes ([batch_size, 223, 303], [batch_size, 55, 74])


### NUMPY DATA

def get_debug_data(): 
	return ("images numpy-array with shape [223, 303, 3]", "depthmaps numpy-array shape [55, 74]")

def get_validation_data():
	return ("images numpy-array with shape [223, 303, 3]", "depthmaps numpy-array shape [55, 74]")

def get_test_data():
	return ("images numpy-array with shape [223, 303, 3]", "depthmaps numpy-array shape [55, 74]")



	






