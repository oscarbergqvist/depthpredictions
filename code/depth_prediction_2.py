'''
"MAIN" module 
All operations are added to the defaultgraph.
Network functions are found in module network_functions_2 
Display graph in tensorboard by opening a new terminal and write "tensorboard --logdir=tensorboard/debug/01/" where 
the last number depends on which directory the current graph is saved in (see line 35 in this module where the 
FileWriter is created). After this, open the local webpage displayed in the terminal (looks something like http://OSCAR-LENOVO-LAPTOP:6006) 
but with your own username.  
'''

import network_functions_2 as nf
import tensorflow as tf
import read_data_2 as rd
#import os
import pathlib
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import images 
with tf.variable_scope("get_batch_placeholders"):
	image_batch_placeholder = rd.image_batch 
	depthmap_groundtruth_batch_placeholder = rd.depthmap_batch

	depthmaps_groundtruth_summary = tf.summary.image("depthmaps_groundtruth", depthmap_groundtruth_batch_placeholder)
with tf.variable_scope("define_networks"):
	# define coarse and fine networks  
	coarse_depthmap_predictions = nf.get_coarse_network(input_placeholder = image_batch_placeholder)
	fine_depthmap_predictions = nf.get_fine_network(input_placeholder  = image_batch_placeholder, 
													coarse_prediction = coarse_depthmap_predictions)
	coarse_predicted_depthmap_summary = tf.summary.image("coarse_depthmap", coarse_depthmap_predictions)
	fine_predicted_depthmap_summary = tf.summary.image("fine_depthmap", fine_depthmap_predictions)

with tf.Session() as session:

	# tensorboard writer CHANGE THE DIR NUMBER EVERY RUN (27 -> 28 -> 29 etc.)
	# tensorboard/* in .gitignore 

	logdir = "01"  


	writer = tf.summary.FileWriter("./tensorboard/debug/" + logdir + "/", session.graph)


	with tf.variable_scope("coarse_optimizer"):
	# compute cost functions
		coarse_cost = nf.get_cost_function(depthmaps_predicted = coarse_depthmap_predictions, 
										depthmaps_groundtruth = depthmap_groundtruth_batch_placeholder,
										name_scope = "coarse_cost_function")
		coarse_cost_summary = tf.summary.scalar("coarse_cost", coarse_cost)
		tf.add_to_collection(name = "coarse_summaries", value = coarse_cost_summary)
		coarse_optimizer = nf.get_coarse_optimizer(coarse_cost)
		tf.add_to_collection(name = "coarse_summaries", value = coarse_predicted_depthmap_summary)
		tf.add_to_collection(name = "coarse_summaries", value = depthmaps_groundtruth_summary)
	with tf.variable_scope("fine_optimizer"):
		fine_cost = nf.get_cost_function(depthmaps_predicted = fine_depthmap_predictions, 
										depthmaps_groundtruth = depthmap_groundtruth_batch_placeholder,
										name_scope = "fine_cost_function")
		fine_cost_summary = tf.summary.scalar("fine_cost", fine_cost)
		tf.add_to_collection(name = "fine_summaries", value = fine_cost_summary)
		fine_optimizer = nf.get_fine_optimizer(fine_cost) 
		tf.add_to_collection(name = "fine_summaries", value = fine_predicted_depthmap_summary)

	# # calculate optimizers
	# optimizer_coarse = nf.get_coarse_optimizer(coarse_cost)	
	# optimizer_fine = nf.get_coarse_optimizer(fine_cost)	

	# initialize variables 
	
	with tf.variable_scope("coarse_optimization"): 
	
		session.run(tf.global_variables_initializer())	
		# optimize networks
		print("optimizing coarse network ...")
		nf.optimize_coarse_network(session = session, 
								writer = writer, 
								optimizer = coarse_optimizer,
								cost = coarse_cost, 
								batch_size = 9, 
								number_of_epochs = 10)	


	with tf.variable_scope("fine_optimization"): 
		print("optimizing fine network ...")
		nf.optimize_fine_network(session = session, 
								writer = writer, 
								optimizer = fine_optimizer,
								cost = fine_cost, 
								batch_size = 9, 
								number_of_epochs = 3)	
	
	
	save_path = "./model/debug/" + logdir + "/" 
	pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 	
	print("saving session to " + save_path + "session")
	saver = tf.train.Saver()
	save_path = saver.save(session, save_path + "session")

	with tf.variable_scope("preformance_evaluation"):
		nf.test_fine_network(
					node = fine_depthmap_predictions,
					saver = saver,
					session = session, 
					depthmaps_groundtruth = depthmap_groundtruth_batch_placeholder,
					load_path = save_path , 
 					file_name = save_path + "metrices_model_" + logdir + ".txt")

	#tf.add_to_collection(name = "fine_summaries", value = fine_predicted_depthmap_summary)
	

		

	# with tf.variable_scope("preformance_evaluation"):
	# 	# evaluate preformance 
	# 	cost, (threshold_1, threshold_2, threshold_3), scale_invariant_rmse  = nf.get_metrics()
	# 	tf.summary.scalar("final")


	writer.close()