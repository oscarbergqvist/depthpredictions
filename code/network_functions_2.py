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
import read_data_2 as rd
import math 
import numpy as np


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
	return weight_tensor 


def new_biases(length): 
	'''
	Args
	- length : integer, desiered length of bias tensor 
	Returns
	- 1d tesnor with length as argument  

	'''
	zero = tf.constant(value = 0.001, dtype = tf.float32, shape = [length])
	bias_tensor = tf.Variable(initial_value = zero, name = 'biases')
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
		return conv_layer, weight_tensor, bias_tensor


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
		return fc_layer, weight_tensor, bias_tensor


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
		cn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = input_placeholder, 
									filter_size = 11,  
									input_channels = 3,
									output_channels = 96,
									stride = 4,
									name_scope = "conv_layer_1")

		s = tf.summary.histogram(name = "summary_w_1", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_1", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)

		cn = max_pool(input_layer = cn, 
					pool_size = 2, 
					name_scope = "max_pool_1")
		cn = add_zero_padding(input_layer = cn, 
					padding_vertical = 3, 
					padding_horizontal = 3, 
					name_scope = "padding_1")

		# 2nd convolutional layer 
		cn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = cn, 
									filter_size = 5,  
									input_channels = 96,
									output_channels = 256,
									stride = 1,
									name_scope = "conv_layer_2")

		s = tf.summary.histogram(name = "summary_w_2", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_2", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)	
		
		cn = max_pool(input_layer = cn, 
					pool_size = 2, 
					name_scope = "max_pool_2")
		cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 2, 
							padding_horizontal = 2,
							name_scope = "padding_2")

		# 3rd convolutional layer
		cn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = cn, 
									filter_size = 3, 
									input_channels = 256,
									output_channels = 384,
									stride = 1,
									name_scope = "conv_layer_3")

		s = tf.summary.histogram(name = "summary_w_3", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_3", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)

		cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 2, 
							padding_horizontal = 2, 
							name_scope = "padding_3")

		# 4th convolutional layer
		cn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = cn, 
									filter_size = 3, 
									input_channels = 384,
									output_channels = 384,
									stride = 1,
									name_scope = "conv_layer_4")

		s = tf.summary.histogram(name = "summary_w_4", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_4", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)		
		
		cn = add_zero_padding(input_layer = cn, 
							padding_vertical = 1, 
							padding_horizontal = 1,
							name_scope = "padding_4")

		# 5th convolutional layer
		cn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = cn, 
									filter_size = 3, 
									input_channels = 384,
									output_channels = 256,
									stride = 1,
									name_scope = "conv_layer_5")

		s = tf.summary.histogram(name = "summary_w_5", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_5", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
	
		cn = max_pool(input_layer = cn, 
					pool_size = 2, 
					name_scope = "max_pool_3")

		# Vectorization
		cn = vectorize_convolutional_layer(input_layer = cn,
										name_scope = "reshape_1")

		# 1st fc layer with dropout (layer 7)
		with tf.variable_scope("dropout"):
			cn = tf.nn.dropout(cn, 0.5)

		cn, weight_tensor, bias_tensor = add_fully_connected_layer(input_layer = cn, 
											input_nodes = 12288, 
											output_nodes = 4096,
											name_scope = "fc_layer_1")

		s = tf.summary.histogram(name = "summary_w_6", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_6", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)

		# 2nd fc layer (layer 8)
		cn, weight_tensor, bias_tensor = add_fully_connected_layer(input_layer = cn, 
											input_nodes = 4096, 
											output_nodes = 4070,
											use_relu = False,
											name_scope = "fc_layer_2")

		s = tf.summary.histogram(name = "summary_w_7", values = weight_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_7", values = bias_tensor)
		tf.add_to_collection(name = "coarse_var_summaries", value = s)

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
	- input_placeholder : [batches, 223, 303, 3] placeholder variable for input  
	- coarse_prediction : [batches, 55, 74, 1]
	- name_scope : string, name scope for coarse network, default: "fine_network" 
	Returns 
	- 4d tensor with shape [batches, height, width, channels]
	'''
	with tf.variable_scope(name_scope):
		input_placeholder = add_zero_padding(input_layer = input_placeholder, 
											padding_vertical = 4, 
											padding_horizontal = 0, 
											name_scope = "padding_1")
		fn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = input_placeholder, 
									filter_size = 9, 
									input_channels = 3,
									output_channels = 63,
									stride = 2,
									name_scope = "conv_layer_1")

		s = tf.summary.histogram(name = "summary_w_1", values = weight_tensor)
		tf.add_to_collection(name = "fine_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_1", values = bias_tensor)
		tf.add_to_collection(name = "fine_var_summaries", value = s)

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
							name_scope = "padding_2")

		# 2nd convolutional layer (FINE NET)
		fn, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = fn, 
									filter_size = 5, 
									input_channels = 64, 
									output_channels = 64,
									stride = 1, 
									name_scope = "conv_layer_2")

		s = tf.summary.histogram(name = "summary_w_2", values = weight_tensor)
		tf.add_to_collection(name = "fine_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_2", values = bias_tensor)
		tf.add_to_collection(name = "fine_var_summaries", value = s)

		fn = add_zero_padding(input_layer = fn, 
							padding_vertical = 4, 
							padding_horizontal = 4, 
							name_scope = "padding_3")

		# 3rd convulutional layer (FINE NET)
		fine_prediction, weight_tensor, bias_tensor = add_convolutional_layer(input_layer = fn, 
									filter_size = 5, 
									input_channels = 64, 
									output_channels = 1,
									stride = 1,
									use_relu = False,
									name_scope = "conv_layer_3")

		s = tf.summary.histogram(name = "summary_w_3", values = weight_tensor)
		tf.add_to_collection(name = "fine_var_summaries", value = s)
		s = tf.summary.histogram(name = "summary_b_3", values = bias_tensor)
		tf.add_to_collection(name = "fine_var_summaries", value = s)

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
		depthmaps_predicted 	= depthmaps_predicted[:, :, :, 0]
		depthmaps_groundtruth 	= depthmaps_groundtruth[:, :, :, 0] 

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

def get_scale_invariant_rmse(depthmaps_predicted, depthmaps_groundtruth, name_scope = "scale_invariant_rmse"):
	'''
	Args:
	- depthmaps_predicted : 4d tensor [batchsize, height, width, 1]
	- depthmaps_groundtruth : 3d tensor [batchsize, height, width]
	Returns: 
	- scale_invariant_rmse	
	'''
	with tf.variable_scope(name_scope):
		depthmaps_predicted = depthmaps_predicted[:, :, :, 0] 
		depthmaps_groundtruth = depthmaps_groundtruth[:, :, :, 0]
		pixels = tf.size(input = depthmaps_predicted[0, :, :], out_type = tf.float32)  
		log_difference = tf.log(depthmaps_predicted) - tf.log(depthmaps_groundtruth)
		op1 = tf.reduce_sum(input_tensor = log_difference, axis = [1, 2])
		op1 = tf.square(op1)
		op2 = tf.square(pixels)
		op2 = tf.divide(1.0, op2)
		batch_regularization = tf.multiply(op1, op2)
		op1 = tf.divide(1.0, pixels)  			
		op2 = tf.norm(tensor = log_difference, axis = [1, 2])
		op2 = tf.square(op2)					
		batch_loss = tf.multiply(op1, op2)
		batch_cost =  batch_loss - batch_regularization
		total_cost = tf.reduce_mean(batch_cost)
		scale_invariant_rmse = tf.sqrt(total_cost)
		return scale_invariant_rmse


def get_threshold(depthmaps_predicted, depthmaps_groundtruth, thr = 1.25, exp = 1, name_scope = "threshold"):
	'''
	The threshold is one of the error measurements 
	Args: 
	- depthmaps_predicted : 4d tensor [batchsize, height, width, 1]
	- depthmaps_groundtruth : 3d tensor [batchsize, height, width]
	- thr (defaults to 1.25)
	- exp (defaults to 1.0)  
	Returns: 
	- the proprtion of features for which the quote between predicted and grundtrouth feature value is lower than thr^exp 
	'''
	with tf.variable_scope(name_scope):
		depthmaps_predicted = depthmaps_predicted[:, :, :, 0] 
		depthmaps_groundtruth = depthmaps_groundtruth[:, :, :, 0]
		features_below_thr = tf.reduce_sum(tf.cast(tf.less(tf.maximum(depthmaps_predicted,\
		 depthmaps_groundtruth), tf.pow(thr, exp)), tf.float32)) 
		features = tf.cast(tf.size(depthmaps_predicted), tf.float32)
		threshold = tf.divide(features_below_thr,  features )
		return threshold


def get_metrics(depthmaps_predicted, depthmaps_groundtruth, lambda_param = 0.5):
	'''
	Returns useful metrics such as threshold och scale invariant RMSE. Logs the metrics to TENSORBOARD
	'''
	cost = get_cost_function(depthmaps_predicted, 
							depthmaps_groundtruth, 
							lambda_param = lambda_param,
							name_scope = "cost_function")
	threshold_1 = get_threshold(depthmaps_predicted, 
								depthmaps_groundtruth, 
								exp = 1, 
								name_scope = "threshold_1")
	threshold_2 = get_threshold(depthmaps_predicted, 
							depthmaps_groundtruth, 
							exp = 2, 
							name_scope = "threshold_2")
	threshold_3 = get_threshold(depthmaps_predicted, 
							depthmaps_groundtruth, 
							exp = 3, 
							name_scope = "threshold_3")	
	scale_invariant_rmse = get_scale_invariant_rmse(depthmaps_predicted, 
													depthmaps_groundtruth, 
													name_scope = "scale_invariant_rmse")	
	s = tf.summary.scalar("threshold_1", threshold_1) 
	tf.add_to_collection(name = "coarse_summaries", value = s)
	s = tf.summary.scalar("threshold_2", threshold_2) 
	tf.add_to_collection(name = "coarse_summaries", value = s)
	s = tf.summary.scalar("threshold_3", threshold_3) 
	tf.add_to_collection(name = "coarse_summaries", value = s)
	s = tf.summary.scalar("scale_invariant_rmse", scale_invariant_rmse)
	tf.add_to_collection(name = "coarse_summaries", value = s)
	s = tf.summary.scalar("cost", cost)
	tf.add_to_collection(name = "coarse_summaries", value = s)

	return cost, (threshold_1, threshold_2, threshold_3), scale_invariant_rmse  


def get_coarse_optimizer(coarse_cost):
	'''
	Returns an operation which optimizes the coarse network with different learningrates for different layers
	Args: 
	- coarse_cost 
	Returns:
	- optimization operation (optimizer)
	'''
	with tf.variable_scope("training_coarse"): 
		coarse_conv_var_list = tf.trainable_variables(scope = "define_networks/coarse_net/conv_layer")
		coarse_fc_var_list = tf.trainable_variables(scope = "define_networks/coarse_net/fc_layer")
		gradients_coarse = tf.gradients(coarse_cost, coarse_conv_var_list + coarse_fc_var_list)

		# opt_coarse_conv = tf.train.AdamOptimizer(learning_rate = 0.00001, momentum = 0.0009,
		# 											name = "optimizer_coarse_conv")
		# opt_coarse_fc = tf.train.AdamOptimizer(learning_rate = 0.00001, momentum = 0.0009,
		# 										name = "optimizer_coarse_fc")
		# print(gradients_coarse)
		opt_coarse_conv = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.9,
													name = "optimizer_coarse_conv")
		opt_coarse_fc = tf.train.MomentumOptimizer(learning_rate = 0.1, momentum = 0.9,
												name = "optimizer_coarse_fc")
		grad_coarse_conv = gradients_coarse[:len(coarse_conv_var_list)]
		grad_coarse_fc = gradients_coarse[len(coarse_conv_var_list):]
		train_op_coarse_conv = opt_coarse_conv.apply_gradients(zip(grad_coarse_conv, coarse_conv_var_list))
		train_op_coarse_fc = opt_coarse_fc.apply_gradients(zip(grad_coarse_fc, coarse_fc_var_list))
		return tf.group(train_op_coarse_conv, train_op_coarse_fc)


def get_fine_optimizer(fine_cost):
	with tf.variable_scope("training_fine"):

		fine_conv_var_1 = tf.trainable_variables(scope = "define_networks/fine_net/conv_layer_1")
		fine_conv_var_2 = tf.trainable_variables(scope = "define_networks/fine_net/conv_layer_2")
		fine_conv_var_3 = tf.trainable_variables(scope = "define_networks/fine_net/conv_layer_3")

		gradients_fine_list = tf.gradients(fine_cost, fine_conv_var_1 + fine_conv_var_2 + fine_conv_var_3)

		opt_fine_1_and_3 = tf.train.MomentumOptimizer(learning_rate = 0.0001, momentum = 0.09,\
			name = "momentum_optimizer_fine_1_and_3")
		opt_fine_2 = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.09,\
			name = "momentum_optimizer_fine_2")

		gradients_fine_1_and_3 = gradients_fine_list[ : len(fine_conv_var_1)] + gradients_fine_list[-len(fine_conv_var_3):]
		gradients_fine_2 = gradients_fine_list[len(fine_conv_var_1) : len(fine_conv_var_1) + len(fine_conv_var_2)] 

		train_op_fine_1_and_3 = opt_fine_1_and_3.apply_gradients(zip(gradients_fine_1_and_3, fine_conv_var_1 + fine_conv_var_3))
		train_op_fine_2 = opt_fine_2.apply_gradients(zip(gradients_fine_2, fine_conv_var_2))
		
		return tf.group(train_op_fine_1_and_3, train_op_fine_2) 


def optimize_coarse_network(session, 
							writer,
							optimizer, 
							cost, 
							batch_size, 
							number_of_epochs):
	'''
	trains the coarse network, has a lot of dependency on read_data
	'''
	(train_images, train_depthmaps) = rd.get_data()
	train_feed_dict = {	
			rd.images_placeholder 		: train_images, 
			rd.labels_placeholder 		: train_depthmaps, 
			rd.batch_size_placeholder 	: batch_size
			}


	# summary_op = tf.summary.merge_all()
	
	summary_op = tf.summary.merge(tf.get_collection("coarse_summaries") + tf.get_collection("coarse_var_summaries"))

	session.run(fetches = rd.iterator.initializer, feed_dict = train_feed_dict)

	number_of_batches = len(train_images) // batch_size  

	for epoch_number in range(number_of_epochs):
		for batch_number in range(number_of_batches):
			(_, cost_value, writable_summary) = session.run([optimizer, cost, summary_op])	
		writer.add_summary(writable_summary, epoch_number) 	
		if True: 
			print("Epoch: ", epoch_number, "coarse cost = ", cost_value)


def optimize_fine_network(session, 
						writer, 
						optimizer,
						cost, 
						batch_size, 
						number_of_epochs):
	'''
	trains the fine network
	''' 
	(train_images, train_depthmaps) = rd.get_data()
	feed_dict = {rd.images_placeholder 		: train_images, 
				rd.labels_placeholder 		: train_depthmaps, 
				rd.batch_size_placeholder 	: batch_size}
	
	summary_op = tf.summary.merge(tf.get_collection("fine_summaries") + tf.get_collection("fine_var_summaries"))

	session.run(fetches = rd.iterator.initializer, feed_dict = feed_dict)
 
	number_of_batches = len(train_images) // batch_size  
	for epoch_number in range(number_of_epochs):
		for batch_number in range(number_of_batches):
			(_, cost_value, writable_summary) = session.run([optimizer, cost, summary_op])
		writer.add_summary(writable_summary, epoch_number) 	
		if True: 
			print("Epoch: ", epoch_number, "fine cost = ", cost_value)



# def test_coarse_network():
	

# 	return None 

def test_fine_network(node ,
					saver ,
					session,
					depthmaps_groundtruth, 

					load_path = "/saved_models/01" , 
 					file_name = "metrices_model_01.txt",
 					test = True):
	
	# Runs the network
	saver.restore(sess = session, save_path = load_path)
	if test:
		(images_test, labels_test) = rd.get_test_data()
		feed_dict = {rd.images_placeholder 		: images_test, 
					rd.labels_placeholder 		: labels_test,
					rd.batch_size_placeholder	: images_test.shape[0]}
		
	else:
		(images_validation, labels_validation) = rd.get_validation_data()
		feed_dict = {rd.images_placeholder 		: images_validation, 
					rd.labels_placeholder 		: labels_validation,
					rd.batch_size_placeholder	: images_validation.shape[0]}
	###TEST###
	
	cost, threshold, scale_invariant_rmse =\
	  get_metrics(node, depthmaps_groundtruth, lambda_param = 0.5)
	session.run(fetches = rd.iterator.initializer, feed_dict = feed_dict)
	metrics_values = session.run(fetches = [cost, threshold, scale_invariant_rmse])

	
	
	# cost_value = metrics_values[0]
	threshold_value = metrics_values[1]

	with open(file_name, "w") as output_file:
		out_string = ""
		out_string += "Cost: " + str(metrics_values[0])
		out_string += ", Threshold1: " + str(threshold_value[0])
		out_string += ", Threshold2: " + str(threshold_value[0])
		out_string += ", Threshold3: " + str(threshold_value[0])
		out_string += ", Scale invariant RMSE: " + str(metrics_values[2])
		output_file.write(out_string)
	
	return None 