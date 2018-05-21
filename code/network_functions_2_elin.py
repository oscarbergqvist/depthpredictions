'''
In this module all the networkrelated functions are specified.
They can be accessed from the outside by importing this module.
Note the variable_scope hirarchy as: 

fine_net
coarse_net
	conv_layer_i
	fc_layer_i 
		weights
		biases
		
This makes it easier to access the variables from any scope and 
results in a nice visualization in tensorboard.
'''


import tensorflow as tf 
import math


def new_weights(shape):
	'''
	Args
	- shape : desiered shape of weights 
	Returns
	- tesnor with shape as argument  

	'''
	weight_tensor = tf.Variable(
			initial_value = tf.truncated_normal(shape = shape, stddev = 0.0001), 
			name = 'weights')
	tf.summary.histogram("W", weight_tensor)
	return weight_tensor 


def new_biases(length): 
	'''
	Args
	- length : integer, desiered length of bias tensor 
	Returns
	- 1d tesnor with length as argument  

	'''
	bias_tensor = tf.Variable(
			initial_value = tf.truncated_normal(shape = [length], stddev = 0.0001), 
			name = 'biases')
	tf.summary.histogram("B", bias_tensor)
	return bias_tensor


def add_convolutional_layer(input_layer, 
							filter_size,
							input_channels,
							output_channels,
							stride = 1,
							use_relu = True,
							name_scope = 'convolutional_layer'):
	''' 
	Params: 
		- input_layer : 4d tensor with shape [batch_size, height, width, channels]  
		- filter size : integer number
		- output_channels : integer number 
		- stride : integer number -> equal stride over height and width  
		- use_relu : boolean, defaults to True   
		- name_scope : string -> name of name scope of the convolutional layer  
	Returns: 
		- 4d tensor with shape [batch_size, height, width, channels]   
	'''
	with tf.variable_scope(name_scope):
		filter_shape = [filter_size, filter_size, input_channels, output_channels]
		weight_tensor = new_weights(shape = filter_shape)
		bias_tensor =	new_biases(length = output_channels)
		conv_layer = tf.nn.conv2d(input = input_layer, 
								filter = weight_tensor, 
								strides = [1, stride, stride, 1],
								padding = "VALID")
		conv_layer = tf.add(conv_layer, bias_tensor)
		if use_relu:
			conv_layer = tf.nn.relu(conv_layer)
		return conv_layer


def add_fully_connected_layer(input_layer, 
							input_nodes, 
							output_nodes,
							use_relu = True,
							name_scope = 'fully_connected_layer'):
	''' 
	Params: 
		- input_layer : 4d tensor with shape [batch_size, height, width, channels]  
		- input_nodes : integer number
		- output_nodes : integer number 
		- use_relu : boolean, defaults to True   
		- name_scope : string -> name of name scope of the convolutional layer  
	Returns: 
		- 4d tensor with shape [batch_size, height, width, channels]   
	'''
	with  tf.variable_scope(name_scope):
		weight_tensor = new_weights(shape = [input_nodes, output_nodes])
		bias_tensor =	new_biases(length = output_nodes)
		fc_layer = tf.add(tf.matmul(input_layer, weight_tensor), bias_tensor)
		if use_relu:
			fc_layer = tf.nn.relu(fc_layer)
		return fc_layer


def vectorize_convolutional_layer(input_layer, name_scope = "reshape"):
	'''
	Params:
	- input_layer : 4d tensor with shape [batch_size, height, width, channels]  
	Returns: 
	- 2d tensor with shape [batch_size, height * width * channels]
	'''
	with tf.variable_scope(name_scope):
		shape = input_layer.get_shape()
		number_of_elements = shape[1:4].num_elements()
		return tf.reshape(input_layer, [-1, number_of_elements]) 


def vector_to_tensor(input_layer, matrix_height, matrix_width, name_scope = "reshape"):
	'''
	Params:
	- input_layer : 2d tensor with shape [batch_size, height * width * channels]
	- matrix_height : integer number, desiered height  
	- matrix_width : integer number, desiered width
	Returns: 
	- input_layer : 4d tensor with shape [batch_size, height, width, channels]   
	'''
	with tf.variable_scope(name_scope):
		return tf.reshape(input_layer, [-1, matrix_height, matrix_width, 1]) 


def max_pool(input_layer, pool_size, name_scope = "max_pool"):
	'''
	Params:
	- input_layer : 4d tensor with shape [batch_size, height, width, channels] 
	- pool_size : size of max_pool 
	Returns: 
	- 2d tensor with shape [batch_size, height * width * channels]
	'''	
	with tf.variable_scope(name_scope):
		return tf.nn.max_pool(value = input_layer,
	    					ksize = [1, pool_size, pool_size, 1],
	    					strides = [1, pool_size, pool_size, 1],
	    					padding = 'VALID')

# Debugged
def add_zero_padding(input_layer, padding_vertical, padding_horizontal, name_scope = "padding"):
	'''
	Params:
	- input_layer : 4d tensor with shape [batch_size, height, width, channels] 
	- padding_vertical : integer number of total padded rows (under + above input) 
	- padding_horizontal : integer number of total padded columns (left + right of input) 
	Returns: 
	- 4d padded tensor
	'''	
	with tf.variable_scope(name_scope):
		padding_vertical_left = math.floor(padding_vertical / 2)  
		padding_vertical_right = math.floor(padding_vertical / 2) + padding_vertical % 2
		padding_horizontal_left = math.floor(padding_horizontal / 2)  
		padding_horizontal_right = math.floor(padding_horizontal / 2) + padding_horizontal % 2
		paddings = tf.constant([[0, 0],
								[padding_vertical_left, padding_vertical_right], 
								[padding_horizontal_left, padding_horizontal_right],
								[0, 0]])
		return tf.pad(tensor = input_layer, paddings = paddings)


def get_coarse_network(input_placeholder, name_scope = "coarse_net"):
	'''
	Returns the coarse network
	Params 
	- input_placeholder : [batches, 223, 303, 3] placeholder variable for input  
	- name_scope : string, name scope for coarse network, default: "coarse_network" 
	Returns 
	- 4d tensor with shape [batches, 55, 74, channels]
	'''
	with tf.variable_scope(name_scope): 
		cn = add_convolutional_layer(input_layer = input_placeholder, 
									filter_size = 11,  
									input_channels = 3,
									output_channels = 96,
									stride = 4,
									name_scope = "conv_layer_1")
		cn = max_pool(input_layer = cn, 
					pool_size = 2, 
					name_scope = "max_pool_1")
		cn = add_zero_padding(input_layer = cn, 
					padding_vertical = 3, 
					padding_horizontal = 3, 
					name_scope = "padding_1")

		# 2nd convolutional layer 
		cn = add_convolutional_layer(input_layer = cn, 
									filter_size = 5,  
									input_channels = 96,
									output_channels = 256,
									stride = 1,
									name_scope = "conv_layer_2")
		cn = max_pool(input_layer = cn, 
					pool_size = 2, 
					name_scope = "max_pool_2")
		cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 2, 
							padding_horizontal = 2,
							name_scope = "padding_2")

		# 3rd convolutional layer
		cn = add_convolutional_layer(input_layer = cn, 
									filter_size = 3, 
									input_channels = 256,
									output_channels = 384,
									stride = 1,
									name_scope = "conv_layer_3")
		cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 2, 
							padding_horizontal = 2, 
							name_scope = "padding_3")

		# 4th convolutional layer
		cn = add_convolutional_layer(input_layer = cn, 
									filter_size = 3, 
									input_channels = 384,
									output_channels = 384,
									stride = 1,
									name_scope = "conv_layer_4")
		cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 1, 
							padding_horizontal = 1,
							name_scope = "padding_4")

		# 5th convolutional layer
		cn = add_convolutional_layer(input_layer = cn, 
									filter_size = 3, 
									input_channels = 384,
									output_channels = 256,
									stride = 1,
									name_scope = "conv_layer_5")
		cn = max_pool(input_layer = cn, 
					pool_size = 2, 
					name_scope = "max_pool_3")

		# Vectorization
		cn = vectorize_convolutional_layer(input_layer = cn,
										name_scope = "reshape_1")

		# 1st fc layer with dropout (layer 7)
		with tf.variable_scope("dropout"):
			cn = tf.nn.dropout(cn, 0.5)

		cn = add_fully_connected_layer(input_layer = cn, 
											input_nodes = 12288, 
											output_nodes = 4096,
											name_scope = "fc_layer_1")

		# 2nd fc layer (layer 8)
		cn = add_fully_connected_layer(input_layer = cn, 
											input_nodes = 4096, 
											output_nodes = 4070,
											use_relu = False,
											name_scope = "fc_layer_2")

		# Vector to tensor 
		coarse_network = vector_to_tensor(input_layer = cn, 
										matrix_height = 55, 
										matrix_width = 74, 
										name_scope = "reshape_2") 
		return coarse_network


def get_fine_network(input_placeholder, coarse_prediction, name_scope = "fine_net"):
	'''
	Returns the fine network
	Params 
	- input_placeholder : [batches, 227, 303, 3] placeholder variable for input  
	- coarse_prediction : [batches, 55, 74, 1]
	- name_scope : string, name scope for coarse network, default: "fine_network" 
	Returns 
	- 4d tensor with shape [batches, height, width, channels]
	'''
	with tf.variable_scope(name_scope):
		fn = add_convolutional_layer(input_layer = input_placeholder, 
									filter_size = 9, 
									input_channels = 3,
									output_channels = 63,
									stride = 2,
									name_scope = "conv_layer_1")
		fn = max_pool(input_layer = fn, 
					pool_size = 2, 
					name_scope = "max_pool_1")

		# Concatination with coarse net
		with tf.variable_scope("concatenation"):
			fn = tf.concat([coarse_prediction, fn], 3)
			fn = tf.nn.relu(fn)
		fn = add_zero_padding(input_layer = fn, 
							padding_vertical = 4, 
							padding_horizontal = 4, 
							name_scope = "padding_1")

		# 2nd convolutional layer (FINE NET)
		fn = add_convolutional_layer(input_layer = fn, 
									filter_size = 5, 
									input_channels = 64, 
									output_channels = 64,
									stride = 1, 
									name_scope = "conv_layer_2")
		fn = add_zero_padding(input_layer = fn, 
							padding_vertical = 4, 
							padding_horizontal = 4, 
							name_scope = "padding_2")

		# 3rd convulutional layer (FINE NET)
		fine_prediction = add_convolutional_layer(input_layer = fn, 
									filter_size = 5, 
									input_channels = 64, 
									output_channels = 1,
									stride = 1,
									use_relu = False,
									name_scope = "conv_layer_3")
		return fine_prediction


def get_cost_function(depthmaps_predicted, 
					depthmaps_groundtruth, 
					lambda_param = 0.5,
					name_scope = "cost_function"):
	'''
	Args:
	- depthmaps_predicted : 4d tensor [batchsize, height, width, 1]
	- depthmaps_groundtruth : 3d tensor [batchsize, height, width]
	- lambda param
	- name_scope 
	Returns: 
	- cost (scalar number)
	'''
	with tf.variable_scope(name_scope):
		depthmaps_predicted = depthmaps_predicted[:, :, :, 0] 
		pixels = tf.size(input = depthmaps_predicted[0, :, :], out_type = tf.float32)  
		log_difference = tf.log(depthmaps_predicted) - tf.log(depthmaps_groundtruth)
		op1 = tf.reduce_sum(input_tensor = log_difference, axis = [1, 2])
		op1 = tf.square(op1)
		op2 = tf.square(pixels)
		op2 = tf.divide(1.0, op2)
		op2 = tf.multiply(tf.constant(lambda_param), op2)
		batch_regularization = tf.multiply(op1, op2)
		op1 = tf.divide(1.0, pixels)  			
		op2 = tf.norm(tensor = log_difference, axis = [1, 2])
		op2 = tf.square(op2)					
		batch_loss = tf.multiply(op1, op2)
		batch_cost =  batch_loss - batch_regularization
		total_cost = tf.reduce_mean(batch_cost)
		return total_cost


def get_coarse_optimizer(coarse_cost):
	'''
	Returns an operation which optimizes the coarse network with different learningrates for different layers
	Args: 
	- coarse_cost 
	Returns:
	- optimization operation (optimizer)
	'''
	with tf.variable_scope("training"): 
		coarse_conv_var_list = tf.trainable_variables(scope = "coarse_net/conv_layer")
		coarse_fc_var_list = tf.trainable_variables(scope = "coarse_net/fc_layer")
		gradients_coarse = tf.gradients(coarse_cost, coarse_conv_var_list + coarse_fc_var_list)
		opt_coarse_conv = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
		opt_coarse_fc = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
		grad_coarse_conv = gradients_coarse[:len(coarse_conv_var_list)]
		grad_coarse_fc = gradients_coarse[len(coarse_conv_var_list):]
		train_op_coarse_conv = opt_coarse_conv.apply_gradients(zip(grad_coarse_conv, coarse_conv_var_list))
		train_op_coarse_fc = opt_coarse_fc.apply_gradients(zip(grad_coarse_fc, coarse_fc_var_list))
		return tf.group(train_op_coarse_conv, train_op_coarse_fc)



def get_fine_optimizer(fine_cost):
	with tf.variable_scope("training_fine"):

		fine_conv_var_1= tf.trainable_variables(scope = "fine_net/conv_layer_1")
		fine_conv_var_2 = tf.trainable_variables(scope = "fine_net/conv_layer_2")
		fine_conv_var_3 = tf.trainable_variables(scope = "fine_net/conv_layer_3")
		opt_fine_1_and_3 = tf.train.MomentumOptimizer(learning_rate = 0.00001, momentum = 0.0009,\
			name = "momentum_optimizer_fine_1_and_3")
		opt_fine_2 = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum = 0.9,\
			name = "momentum_optimizer_fine_2")
		# opt_fine_1_and_3 = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.009,\
		# 	name = "adam_optimizer_fine_1_and_3")
		# opt_fine_2 = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.9,\
		# 	name = "adam_optimizer_fine_2")
		
		gradients_fine_1_and_3 = opt_fine_1_and_3.compute_gradients(loss = fine_cost, var_list = fine_conv_var_1 + fine_conv_var_3)
		gradients_fine_2 = opt_fine_2.compute_gradients(loss = fine_cost, var_list = fine_conv_var_2)
		
		train_op_fine_1_and_3 = opt_fine_1_and_3.apply_gradients(gradients_fine_1_and_3)
		train_op_fine_2 = opt_fine_2.apply_gradients(gradients_fine_2)
		
		return tf.group(train_op_fine_1_and_3, train_op_fine_2) 







# def optimize(session, optimizer, batch_size, number_of_epochs):
# 	(img_batch_list, depthmap_groundtruth_list) = data.get_batch_lists(batch_size)
# 	start_time = time.time()
# 	for epoch_number in range(number_of_epochs):
# 		for batch in range(img_batch_list):
# 			session.run(iterator.initializer, 
# 						feed_dict = {images_placeholder:images, depths_placeholder: depths})

# 			(batch, depthmap_groundtruth) = data.get_batch(batch_number, batch_size)
# 			feed_dict = {b}







