'''
"MAIN" module 
All operations are added to the defaultgraph.
Network functions are found in module network_functions_2 
Display graph in tensorboard by opening a new terminal and write "tensorboard --logdir=tensorbaord/debug/01/" where 
the last number depends on which directory the current graph is saved in (see line 35 in this module where the 
FileWriter is created). After this, open the local webpage displayed in the terminal (looks something like http://OSCAR-LENOVO-LAPTOP:6006) 
but with your own username.  
'''

import network_functions_2_elin as nf
import tensorflow as tf
import numpy as np
import read_data as rd


with tf.name_scope("input_data"):
	# import images 
	(iterate_data, sub_images, sub_depths, sub_images_placeholder, sub_depths_placeholder) = rd.read_debug_data()	
	sub_images_coarse = tf.constant(value = np.moveaxis(sub_images[0:223, 0:303, :, :], -1, 0), dtype = tf.float32, name = "images_coarse") 
	sub_images_fine = tf.constant(value = np.moveaxis(sub_images[0:227, 0:303, :, :], -1, 0), dtype = tf.float32, name = "images_fine") 
	depthmaps_groundtruth = tf.constant(value = np.moveaxis(sub_depths[0:55, 0:74, :], -1, 0), dtype = tf.float32, name = "depthmaps_groundtruth")

	sub_images_coarse = tf.constant(value = sub_images[:,0:223, 0:303, :], dtype = tf.float32, name = "images_coarse") 
	sub_images_fine = tf.constant(value = sub_images[:, 0:227, 0:303, :], dtype = tf.float32, name = "images_fine") 
	depthmaps_groundtruth = tf.constant(value = np.moveaxis(sub_depths[:,0:55, 0:74, :], -1, 0), dtype = tf.float32, name = "depthmaps_groundtruth")
	
	# print sample images to tensorboard 
	tf.summary.image(name = "images_coarse", tensor = sub_images_coarse, max_outputs = 1)
	tf.summary.image(name = "images_fine", tensor = sub_images_fine, max_outputs = 1)


# define coarse and fine networks  
coarse_depthmap_predictions = nf.get_coarse_network(input_placeholder = sub_images_coarse)
fine_depthmap_predictions = nf.get_fine_network(input_placeholder  = sub_images_fine, coarse_prediction = coarse_depthmap_predictions)


# Session: tensorflow calculates all values using the input 
with tf.Session() as sess:

	# tensorboard writer CHANGE THE DIR NUMBER EVERY RUN (27 -> 28 -> 29 etc.)
	# tensorboard/* in .gitignore 
	writer = tf.summary.FileWriter("./tensorboard/debug/07", sess.graph) 	

	sess.run(tf.global_variables_initializer())	
							 
	sess.run(fine_depthmap_predictions)										

	# compute cost function 
	fine_cost = nf.get_cost_function(depthmaps_predicted = fine_depthmap_predictions, 
									depthmaps_groundtruth = depthmaps_groundtruth)

	# calculate and run optimizer 
	optimizer_fine = nf.get_fine_optimizer(fine_cost)	
	sess.run(tf.global_variables_initializer())			
	sess.run(optimizer_fine)

	# this code makes sure that all info gets written to tensorboard 
	merged_summary = sess.run(tf.summary.merge_all())
	writer.add_summary(merged_summary)
	writer.close()


	
