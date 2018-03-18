"""
LSTM TensorFlow Assignments
Advanced Topics in Machine Learning, UCL (COMPGI13)
Coursework 2
Load models and validate them on training & validation sets
Author: Adam Hornsby
"""

# native python functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import itertools
import sys
import copy

# tensorflow data processing functions
from tensorflow.examples.tutorials.mnist import input_data

# joblib for dumping and loading model classes
import joblib

# matplotlib for plotting
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# own functions
import vis_funcs
import log_funcs
from run_lstm import Exercise1_config, data_config, build_model_graph, accuracy, categorical_crossentropy, preprocess_data

OUTPUT_PATH = '../output/'
BATCH_SIZE = 200

EXERCISE_1_MODELS = {
	
		'LSTM_32' : {'file' : 'ex1_LSTM_Linear_32_100_session.ckpt', 'cells' : ['LSTM', 'Linear'] , 'neurons' : [32, 100]}
	,	'LSTM_64' : {'file' : 'ex1_LSTM_Linear_64_100_session.ckpt', 'cells' : ['LSTM', 'Linear'], 'neurons' : [64, 100]}
	,	'LSTM_128': {'file' : 'ex1_LSTM_Linear_128_100_session.ckpt', 'cells' : ['LSTM', 'Linear'], 'neurons' : [128, 100]}
	,	'StackedLSTM_32' : {'file' : 'ex1_StackedLSTM_Linear_32_100_session.ckpt', 'cells' : ['StackedLSTM', 'Linear'], 'neurons' : [32, 100]} 

	,	'GRU_32' : {'file' : 'ex1_GRU_Linear_32_100_session.ckpt', 'cells' : ['GRU', 'Linear'], 'neurons' : [32, 100]}
	,	'GRU_64' : {'file' : 'ex1_GRU_Linear_64_100_session.ckpt', 'cells' : ['GRU', 'Linear'], 'neurons' : [64, 100]}
	, 	'GRU_128': {'file' : 'ex1_GRU_Linear_128_100_session.ckpt', 'cells' : ['GRU', 'Linear'], 'neurons' : [128, 100]}
	,	'StackedGRU_32' : {'file' : 'ex1_StackedGRU_Linear_32_100_session.ckpt', 'cells' : ['StackedGRU', 'Linear'], 'neurons' : [32,100]}

}

EXERCISE_2_MODELS = {

		'GRU_32' : {'file' : 'ex2_GRU_32_session.ckpt', 'cells' : ['GRU'], 'neurons' : [32]}
	,	'GRU_64' : {'file' : 'ex2_GRU_64_session.ckpt', 'cells' : ['GRU'], 'neurons' : [64]}
	, 	'GRU_128': {'file' : 'ex2_GRU_128_session.ckpt', 'cells' : ['GRU'], 'neurons' : [128]}
	,	'StackedGRU_32' : {'file' : 'ex2_StackedGRU_32_session.ckpt', 'cells' : ['StackedGRU'], 'neurons' : [32]}

}

def batch_loss_and_acc(mnist_set, batch_size, exercise=1):
	"""Score up a dataset by batch and then return the average acc and loss"""

	acc = 0
	loss = 0
	num_examples = mnist_set.num_examples
	total_batches = int(num_examples/batch_size)

	for _ in range(total_batches):

		train_xs, train_ys = mnist_set.next_batch(batch_size)
		train_xs, train_ys = preprocess_data(train_xs, train_ys, max_time=data_config['num_steps'], exercise=exercise)
		
		if exercise == 1:
			tmp_acc, tmp_loss = sess.run([acc_eval, mse_eval], feed_dict={x: train_xs, y_: train_ys})
		elif exercise == 2:
			tmp_loss = sess.run(seq_loss, feed_dict={x: train_xs, y_: train_ys})
			tmp_loss = tmp_loss
			tmp_acc = 0

		acc += tmp_acc 
		loss += tmp_loss

	acc /= float(total_batches)
	loss /= float(total_batches)

	return acc, loss

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print('usage python load_models.py exercise_number')

	else:
		exercise = int(sys.argv[1])
		# import dataset with one-hot class encoding
		mnist = input_data.read_data_sets('../Coursework_1/input_data/', one_hot=True)

		if exercise == 1:
			model_info = EXERCISE_1_MODELS
		elif exercise == 2:
			model_info = EXERCISE_2_MODELS

		for model_name in model_info.keys():

			tf.reset_default_graph()

			ckpt_path = OUTPUT_PATH + EXERCISE_1_MODELS[model_name]['file']
			hidden_neurons = EXERCISE_1_MODELS[model_name]['neurons']
			cell_types = EXERCISE_1_MODELS[model_name]['cells']

			model_config = Exercise1_config(hidden_neurons=hidden_neurons, cell_type=cell_types)

			# build tensorflow graph
			train_step, x, y_, y, train_state, init_state, summary = build_model_graph(model_name
								, model_config=model_config
								, data_config=data_config
								, exercise=exercise
								)

			init_op = tf.global_variables_initializer()
			saver = tf.train.Saver()

			with tf.Session() as sess:
				sess.run(init_op)
				saver.restore(sess, ckpt_path)

				seq_loss = tf.contrib.seq2seq.sequence_loss(y, y_, tf.ones([BATCH_SIZE, 783]))
				train_acc, train_loss = batch_loss_and_acc(mnist.train, BATCH_SIZE, exercise=exercise)
				test_acc, test_loss = batch_loss_and_acc(mnist.test, BATCH_SIZE, exercise=exercise)

				print('Performance of model {0:s}'.format(model_name))
				print('Accuracy - Train: {0:f} Test: {1:f}'.format(train_acc, test_acc))
				print('Crossentropy Loss - Train: {0:f} Test: {1:f}' .format(train_loss, test_loss))

