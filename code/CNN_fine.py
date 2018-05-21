import tensorflow as tf
import numpy as np
import math
import read_data as da

# Fine 1
filter_size1 = 9 	# Here we want a 9x9 pixels filter
num_filters1 = 63 	# Here we want 63 filters

# Fine 2
## In this layer we only want to concatenate the output from fine1 with
## output from coarse7

# Fine 3
filter_size3 = 5 	# Here we want a 5x5 pixels filter
num_filters3 = 64	# Here we want 64 filters


# Fine 4
filter_size4 = 5 	# Here we want a 5x5 pixels filter
num_filters4 = 1	# Here we want 64 filters

#Data dimensions
# The real input is a RGB image with the dimension 304x228x3. but when one
# recalculate the size from the network the size becomes 303x227x3.

#NYUDepth
input_sizes_width =  303
input_sizes_height = 227
input_depth =  3

output_size_width = 74
output_size_height = 55
output_depth = 1

#KITTI
# input_sizes_width =  576
# input_sizes_height = 172
# input_depth =  3

# output_size_width = 142
# output_size_height = 27
# output_depth = 1


# Help functions for creating new variables

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))


# Help functions for creating convolutional layer

def new_conv_layer(input, num_input_channels, filter_size,\
 num_filters, use_pooling=True, strides_in = [1, 1, 1, 1], padding_in = 'SAME'):

	shape = [filter_size, filter_size, num_input_channels, num_filters]

	weights = new_weights(shape=shape)

	biases = new_biases(length=num_filters)

	# create the convolution layer

	layer = tf.nn.conv2d(input = input, filter=weights, strides = strides_in,\
		padding = padding_in)
	layer += biases

	if use_pooling:
		layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1],\
		padding = 'SAME' )

	layer = tf.nn.relu(layer)

	return layer, weights

#Placeholder variables

x_image = tf.placeholder(tf.float32, shape=[None, input_sizes_height,\
 input_sizes_width, input_depth])

output_coarse = tf.placeholder(tf.float32, shape=[None, 55,\
 74, 1])

#Fine1

fine1, weights_fine1 = new_conv_layer(input=x_image, num_input_channels = \
	input_depth	,filter_size = filter_size1, num_filters = num_filters1, \
	use_pooling = True,	strides_in = [1,  2, 2, 1], padding_in = 'VALID') #I think VALID padding here is correct

#Fine2

fine2 = tf.concat([fine1, output_coarse], 3) #I hope dimension 3 is correct

#Fine3

fine3, weights_fine3 = new_conv_layer(input=fine2, num_input_channels =\
 num_filters1+1	,filter_size = filter_size3, num_filters = num_filters3, \
 use_pooling = False, strides_in = [1,  1, 1, 1], padding_in = 'SAME')

#Fine4

fine4, weights_fine4 = new_conv_layer(input=fine3, num_input_channels = \
	num_filters3 ,filter_size = filter_size4, num_filters = num_filters4,\
	 use_pooling = False, strides_in = [1,  1, 1, 1], padding_in = 'SAME')


# def cost_function(prediction, true_output, lambdaa=0.5): #some problem with a none int
# 	#OBS!!! Tar in en bild!!!!
# 	#tf.log Computes natural logarithm of x element-wise.
# 	#dimensions of prediction 74x55 or 142x27
# 	n = tf.size(prediction, out_type = tf.float64)
# 	lambdaa = tf.constant(lambdaa, dtype=tf.float64)
# 	d_i = tf.log(true_output) - tf.log(prediction)
# 	loss_term1 = tf.multiply(tf.divide(1.0,n),tf.reduce_sum(tf.square(d_i)))
# 	loss_term2 = tf.multiply(tf.divide(lambdaa,tf.square(n)),  \
# 		tf.square(tf.reduce_sum(d_i)))
# 	loss = tf.subtract(loss_term1,loss_term2)

# 	return loss

def cost_function(depthmap_predicted, depthmap_groundtruth, regularization_param):
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


# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# def train(dataset,sub_images, sub_depths, number_epochs = 10, \
	# batch_size = 3, lambdaa=0.5 , sess = tf.Session()):
	# The training of the netwo
		# dataset is a object of the class tf.data.Dataset
			# and is a zip version of input_pictures and 
			# groundtruth of lables. It contains the placeholders
		# number of epochs is the number of epochs
		# batch_size is the number of elements in a batch
		# lambdaa is the regularization parameter from equation (4) 
			# in th paper
		# sess is the started session, by default it is initialized
			# when the function is called

	
	# loss = tf.constant(0.0, shape = [number_epochs,1])


	# dataset =  dataset.batch(batch_size)
	# iterator = dataset.repeat(number_epochs)
	# iterator = dataset.make_initializable_iterator()
	# sess.run(iterator.initializer, feed_dict={sub_images_placeholder: features,
                                          # labels_placeholder: labels}))

	# for _ in range(number_epochs):
  	
 #  		while True:
 #    		try:
 #      			next_example, next_label = iterator.get_next()
 #    		except tf.errors.OutOfRangeError:
 #      			break

 #      	loss = cost_function()


## dataset.batch(n) n is the number of element in each batch
## iterator.get_next() gets the next batch





# test = tf.constant([[1, 2], [3, 4]], dtype=tf.float64)
# test2 = tf.constant([[1, 1],[1, 1]], dtype=tf.float64)
test = tf.data.Dataset.from_tensor_slices([[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]])
test2 = tf.data.Dataset.from_tensor_slices([[-1.0,-1.0, -1.0, -1.0], [0.0 ,0.0, 0.0, 0.0]])
dataset = tf.data.Dataset.from_tensor_slices((test, test2))

# dataset = tf.data.Dataset.zip((test, test2))
dataset = dataset.repeat(4)

batches =  dataset.batch(4)
#batches= batches.shuffle(buffer_size=3)
#iterator = batches.make_one_shot_iterator()
iterator = batches.make_initializable_iterator()
sess = tf.Session()
sess.run(iterator.initializer)

next_element = iterator.get_next()

# test_padding = tf.constant([[ 1, 0],[0 ,0]])
result = sess.run(next_element)
print(result[0])

result = sess.run(next_element)
print(result)

# result = sess.run(next_element)
# print(result[0], result[1])

# result = sess.run(next_element)
# print(result[0], result[1])

# result = sess.run(next_element)
# print(result[0],result[1])
# msg = "Optimization Iteration: {0:>6}, Training Accuracy: {2:>6.1%}"
# acc = 10
# # Print it.
# print(msg.format(12 + 1, acc, 4))

























# n_of_nodes = 3
# output_dimension = 1
# batch_size = 100
# n_epoch = 10

# input_s = tf.placeholder('float', [None,3])
# output = tf.placeholder('float')



# def neural_network_model(data):
# 	#Feed-forward part of the network

# 	# Create the form of the fc layer
# 	output_layer = {'weights':tf.Variable(tf.random.normal([3, output_dimension]\
# 		)), 'biases':tf.Variable(tf.random.normal(output_dimension))}
# 	#calculate the output of the fc layer
# 	output = tf.add(tf.matmul(data, output_layer['weights']),\
# 	 output_layer['biases'])

# 	return output

# def train_neural_network(input_s):
# 	prediction = neural_network_model(input_s)

# test = tf.constant([[1, 2], [3, 4]])
# sess = tf.Session()
# test_padding = tf.constant([[ 1, 0],[0 ,0]])
# result = sess.run(tf.pad(test, test_padding))
# print(result)


			







