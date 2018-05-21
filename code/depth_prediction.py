import network_functions as nf
import tensorflow as tf
import numpy as np
import read_data as rd


(iterate_data, sub_images, sub_depths, sub_images_placeholder, sub_depths_placeholder) = rd.read_debug_data()	
sub_images_course = np.moveaxis(sub_images[0:303, 0:223, :, :], -1, 0) 
sub_images_fine = np.moveaxis(sub_images[0:303, 0:227, :, :], -1, 0) 

image_coarse = tf.Variable(tf.constant(value = sub_images_course, dtype = tf.float32))    	 
image_fine 	 = tf.Variable(tf.constant(value = sub_images_fine, dtype = tf.float32))    	 



# import data, placeholders and iterator 

(iterate_data, sub_images, sub_depths, sub_images_placeholder, sub_depths_placeholder) = rd.read_debug_data()	



(coarse_net, _) = nf.add_convolutional_layer(input_layer = image_coarse, 
							filter_size = 11, 
							input_channels = 3, 
							output_channels = 96,
							stride = 4)

coarse_net = nf.relu(coarse_net)

coarse_net = nf.max_pool(input_layer = coarse_net, 
						pool_size = 2)

coarse_net = nf.add_zero_padding(input_layer = coarse_net, 
							padding_vertical = 3, 
							padding_horizontal = 3)

# 2nd convolutional layer 

(coarse_net, _) = nf.add_convolutional_layer(input_layer = coarse_net, 
							filter_size = 5, 
							input_channels = 96, 
							output_channels = 256,
							stride = 1)

coarse_net = nf.relu(coarse_net)

coarse_net = nf.max_pool(input_layer = coarse_net, 
						pool_size = 2)

coarse_net = nf.add_zero_padding(input_layer = coarse_net, 
							padding_vertical = 2, 
							padding_horizontal = 2)

# 3rd convolutional layer

(coarse_net, _) = nf.add_convolutional_layer(input_layer = coarse_net, 
							filter_size = 3, 
							input_channels = 256, 
							output_channels = 384,
							stride = 1)

coarse_net = nf.relu(coarse_net)

coarse_net = nf.add_zero_padding(input_layer = coarse_net, 
							padding_vertical = 2, 
							padding_horizontal = 2)

# 4th convolutional layer

(coarse_net, _) = nf.add_convolutional_layer(input_layer = coarse_net, 
							filter_size = 3, 
							input_channels = 384, 
							output_channels = 384,
							stride = 1)

coarse_net = nf.relu(coarse_net)

coarse_net = nf.add_zero_padding(input_layer = coarse_net, 
							padding_vertical = 1, 
							padding_horizontal = 1)

# 5th convolutional layer

(coarse_net, _) = nf.add_convolutional_layer(input_layer = coarse_net, 
							filter_size = 3, 
							input_channels = 384, 
							output_channels = 256,
							stride = 1)

coarse_net = nf.relu(coarse_net)

coarse_net = nf.max_pool(input_layer = coarse_net, 
						pool_size = 2)

# Vectorization

coarse_net = nf.vectorize_convolutional_layer(input_layer = coarse_net)

# 1st fc layer with dropout

coarse_net = nf.dropout(coarse_net)

coarse_net = nf.add_fully_connected_layer(input_layer = coarse_net, 
									input_nodes = 12288, 
									output_nodes = 4096)

# 2nd fc layer

coarse_net = nf.add_fully_connected_layer(input_layer = coarse_net, 
									input_nodes = 4096, 
									output_nodes = 4070)
# Vector to tensor 

coarse_net = nf.vector_to_tensor(input_layer = coarse_net, 
								matrix_height = 74, 
								matrix_width = 55)


#################### FINE NET #########################

# 1st convolutional layer (FINE NET)

(fine_net, _) = nf.add_convolutional_layer(input_layer = image_fine, 
							filter_size = 9, 
							input_channels = 3, 
							output_channels = 63,
							stride = 2)

fine_net = nf.relu(fine_net)

fine_net = nf.max_pool(input_layer = fine_net, 
						pool_size = 2)


# Concatination with coarse net

fine_net = nf.concatenate(coarse_net, fine_net)

fine_net = nf.relu(fine_net)

fine_net = nf.add_zero_padding(input_layer = fine_net, 
							padding_vertical = 4, 
							padding_horizontal = 4)

# 2nd convolutional layer (FINE NET)

(fine_net, _) = nf.add_convolutional_layer(input_layer = fine_net, 
							filter_size = 5, 
							input_channels = 64, 
							output_channels = 64,
							stride = 1)

fine_net = nf.relu(fine_net)

fine_net = nf.add_zero_padding(input_layer = fine_net, 
							padding_vertical = 4, 
							padding_horizontal = 4)

# 3rd convulutional layer (FINE NET)

(fine_net, _) = nf.add_convolutional_layer(input_layer = fine_net, 
							filter_size = 5, 
							input_channels = 64, 
							output_channels = 1,
							stride = 1)


writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# session = tf.InteractiveSession()
# session.run(tf.global_variables_initializer())
# print(coarse_net.eval().shape)
# print(fine_net.eval().shape)