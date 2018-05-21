import tensorflow as tf 
import math
import time

# Debugged
def create_weight_tensor(shape, stddev = 0.001):
	return tf.Variable(tf.truncated_normal(shape = shape, stddev = stddev))

# Debugged 
def create_bias_tensor(length, stddev = 0.001):
	return tf.Variable(tf.truncated_normal(shape = [length], stddev = stddev))

# Debugged
def add_convolutional_layer(input_layer, 
							filter_size, 
							input_channels, 
							output_channels,
							stride):
	shape = [filter_size, filter_size, input_channels, output_channels]
	weight_tensor = create_weight_tensor(shape = shape)
	bias_tensor = create_bias_tensor(length = output_channels)
	convolutional_layer = tf.nn.conv2d(input = input_layer, 
									filter = weight_tensor, 
									strides = [1, stride, stride, 1],
									padding = "VALID")
	convolutional_layer = tf.add(convolutional_layer, bias_tensor)
	variable_dict = {'w' : weight_tensor, 'b' : bias_tensor}
	return convolutional_layer, variable_dict

# Debugged
def add_fully_connected_layer(input_layer, input_nodes, output_nodes):
	weight_tensor = create_weight_tensor(shape = [input_nodes, output_nodes])
	bias_tensor = create_bias_tensor(length = output_nodes)
	fully_connected_layer = tf.add(tf.matmul(input_layer, weight_tensor), bias_tensor)
	variable_dict = {'w' : weight_tensor, 'bias_tensor' : bias_tensor}
	return fully_connected_layer, variable_dict

# Debugged
def vectorize_convolutional_layer(input_layer):
	shape = input_layer.get_shape()
	number_of_elements = shape[1:4].num_elements()
	return tf.reshape(input_layer, [-1, number_of_elements]) 


def vector_to_tensor(input_layer, matrix_height, matrix_width):
	return tf.reshape(input_layer, [-1, matrix_height, matrix_width, 1]) 

# Debugged
def max_pool(input_layer, pool_size):
	return tf.nn.max_pool(value = input_layer,
    					ksize = [1, pool_size, pool_size, 1],
    					strides = [1, pool_size, pool_size, 1],
    					padding = 'VALID')

# Debugged
def concatenate(coarse_layer, fine_layer):	
	return tf.concat([coarse_layer, fine_layer], 3)

# Debugged 
def relu(input_layer):
	return tf.nn.relu(input_layer)

# Debugged
def add_zero_padding(input_layer, padding_vertical, padding_horizontal):
	padding_vertical_left = math.floor(padding_vertical / 2)  
	padding_vertical_right = math.floor(padding_vertical / 2) + padding_vertical % 2
	padding_horizontal_left = math.floor(padding_horizontal / 2)  
	padding_horizontal_right = math.floor(padding_horizontal / 2) + padding_horizontal % 2
	paddings = tf.constant([[0, 0],
							[padding_vertical_left, padding_vertical_right], 
							[padding_horizontal_left, padding_horizontal_right],
							[0, 0]])
	return tf.pad(tensor = input_layer, paddings = paddings)


def dropout(input_layer):
	return tf.nn.dropout(input_layer, 0.5)


def get_cost_function(depthmap_predicted, depthmap_groundtruth, regularization_param):
	depthmap_predicted = depthmap_predicted[:, :, :, 0]
	depthmap_groundtruth = depthmap_groundtruth[:, :, :, 0]
	pixels = tf.size(input = depthmap_predicted[0, :, :], out_type = tf.float32)  
	log_difference = tf.log(depthmap_predicted) - tf.log(depthmap_groundtruth)
	op1 = tf.reduce_sum(input_tensor = log_difference, axis = [1, 2])
	op1 = tf.square(op1)
	op2 = tf.square(pixels)
	op2 = tf.divide(1.0, op2)
	op2 = tf.multiply(tf.constant(regularization_param), op2)
	batch_regularization = tf.multiply(op1, op2)
	op1 = tf.divide(1.0, pixels)  			
	op2 = tf.norm(tensor = log_difference, axis = [1, 2])
	op2 = tf.square(op2)					
	batch_loss = tf.multiply(op1, op2)
	batch_cost =  batch_loss - batch_regularization
	total_cost = tf.reduce_mean(batch_cost)
	return total_cost

def get_optimizer(cost_function, learning_rate):
	return tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_function)

def optimize(session, optimizer, iterator, dataset, images_placeholder, batch_size, depths_placeholder, number_of_epochs):
	(img_batch_list, depthmap_groundtruth_list) = data.get_batch_lists(batch_size)
	start_time = time.time()
	for epoch_number in range(number_of_epochs):
		for batch in range(img_batch_list):
			session.run(iterator.initializer, 
						feed_dict = {images_placeholder:images, depths_placeholder: depths})

			(batch, depthmap_groundtruth) = data.get_batch(batch_number, batch_size)
			feed_dict = {b}

def get_coarse_network(input):
	(coarse_net_1, coarse_variables_1) = add_convolutional_layer(input_layer = image_coarse_placeholder, 
								filter_size = 11, 
								input_channels = 3, 
								output_channels = 96,
								stride = 4)

	cn = relu(coarse_net_1)

	cn = max_pool(input_layer = cn, 
					pool_size = 2)

	cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 3, 
							padding_horizontal = 3)

	# 2nd convolutional layer 

	(coarse_net_2, coarse_variables_2) = add_convolutional_layer(input_layer = cn, 
								filter_size = 5, 
								input_channels = 96, 
								output_channels = 256,
								stride = 1)

	cn = relu(coarse_net_2)

	cn = max_pool(input_layer = cn, 
					pool_size = 2)

	cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 2, 
							padding_horizontal = 2)

	# 3rd convolutional layer

	(coarse_net_3, coarse_variables_3) = add_convolutional_layer(input_layer = cn, 
								filter_size = 3, 
								input_channels = 256, 
								output_channels = 384,
								stride = 1)

	cn = relu(coarse_net_3)

	cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 2, 
							padding_horizontal = 2)

	# 4th convolutional layer

	(coarse_net_4, coarse_variables_4) = add_convolutional_layer(input_layer = cn, 
								filter_size = 3, 
								input_channels = 384, 
								output_channels = 384,
								stride = 1)

	cn = relu(coarse_net_4)

	cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 1, 
							padding_horizontal = 1)

	# 5th convolutional layer

	(coarse_net_5, coarse_variables_5) = add_convolutional_layer(input_layer = cn, 
								filter_size = 3, 
								input_channels = 384, 
								output_channels = 256,
								stride = 1)

	cn = relu(coarse_net_5)
	cn = max_pool(input_layer = cn, 
					pool_size = 2)

	# Vectorization
	cn = vectorize_convolutional_layer(input_layer = cn)

	# 1st fc layer with dropout (layer 7)
	cn = dropout(cn)
	(coarse_net_6, coarse_variables_6) = add_fully_connected_layer(input_layer = cn, 
										input_nodes = 12288, 
										output_nodes = 4096)

	# 2nd fc layer (layer 8)
	(coarse_net_7, coarse_variables_7) = add_fully_connected_layer(input_layer = coarse_net_6, 
										input_nodes = 4096, 
										output_nodes = 4070)

	# Vector to tensor 
	coarse_network = vector_to_tensor(input_layer = coarse_net_7, 
									matrix_height = 74, 
									matrix_width = 55)

	coarse_variables = {'1' : coarse_variables_1, 
						'2' : coarse_variables_2, 
						'3' : coarse_variables_3, 
						'4' : coarse_variables_4, 
						'5' : coarse_variables_5, 
						'6' : coarse_variables_6,
						'7' : coarse_variables_7}  

	return coarse_network, coarse_variables





def get_fine_network(input):

	(fine_net_1, fine_variables_1) = add_convolutional_layer(input_layer = image_fine_placeholder, 
								filter_size = 9, 
								input_channels = 3, 
								output_channels = 63,
								stride = 2)

	fn = relu(fine_net_1)

	fn = max_pool(input_layer = fn, 
					pool_size = 2)


	# Concatination with coarse net

	fn = concatenate(coarse_prediction, fn)

	fn = relu(fn)

	fn = add_zero_padding(input_layer = fn, 
							padding_vertical = 4, 
							padding_horizontal = 4)

	# 2nd convolutional layer (FINE NET)

	(fine_net_2, fine_variables_2) = add_convolutional_layer(input_layer = fn, 
								filter_size = 5, 
								input_channels = 64, 
								output_channels = 64,
								stride = 1)

	fn = relu(fine_net_2)

	fn = add_zero_padding(input_layer = fn, 
							padding_vertical = 4, 
							padding_horizontal = 4)

	# 3rd convulutional layer (FINE NET)

	(fine_prediction, fine_variables_3) = add_convolutional_layer(input_layer = fn, 
								filter_size = 5, 
								input_channels = 64, 
								output_channels = 1,
								stride = 1)

	fine_variables = {'1' : fine_variables_1, '2' : fine_variables_2, '3' : fine_variables_3 }

	return fine_prediction, fine_variables



# # Load the training data into two NumPy arrays, for example using `np.load()`.
# with np.load("/var/data/training_data.npy") as data:
#   features = data["features"]
#   labels = data["labels"]

# # Assume that each row of `features` corresponds to the same row as `labels`.
# assert features.shape[0] == labels.shape[0]

# features_placeholder = tf.placeholder(features.dtype, features.shape)
# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# # [Other transformations on `dataset`...]
# dataset = ...
# iterator = dataset.make_initializable_iterator()

# sess.run(iterator.initializer, feed_dict={features_placeholder: features,
#                                           labels_placeholder: labels})


# filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(...)
# dataset = dataset.shuffle(buffer_size = 10000)
# dataset = dataset.batch(batch_size)
# dataset = dataset.repeat(number_of_epochs)




