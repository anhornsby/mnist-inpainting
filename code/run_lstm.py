"""
LSTM TensorFlow Assignments
Advanced Topics in Machine Learning, UCL (COMPGI13)
Coursework 2
Neural Network Classes
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

# own functions
import vis_funcs
import log_funcs

# random seed
SEED = 42

# run config
flags = tf.app.flags

flags.DEFINE_string("mnist_path", '../Coursework_1/input_data/', "Directory to read MNIST data.")
flags.DEFINE_string("save_path", '../output/', "Directory to write the model.")
flags.DEFINE_integer("epochs", 15, "How many training epochs?")
flags.DEFINE_integer("batch_size", 200, "Rows to train in each batch")
flags.DEFINE_integer("num_steps", 784, "Number of steps to take during 'truncated' forward/back prop (max of 784)")
flags.DEFINE_integer("exercise", 1, "Which exercise do you want to run models for?")

FLAGS = flags.FLAGS

# DATA CONFIG
# mnist image dimensions
data_config = dict()
data_config = {
	  'image_width' : 28
	, 'image_height' : 28
	, 'input_size' : 5
	, 'output_neurons' : 10
	, 'num_steps' : 784
}

Q1_HyperParameters = zip([['LSTM', 'Linear'], ['LSTM', 'Linear'], ['LSTM', 'Linear'], ['GRU', 'Linear'], ['GRU', 'Linear'], ['GRU', 'Linear'], ['StackedLSTM', 'Linear'], ['StackedGRU', 'Linear']], \
										[[32, 100], [64, 100], [128, 100], [32, 100], [64, 100], [128, 100], [32, 100], [32, 100]])

Q2_HyperParameters = zip([['GRU'], ['GRU'], ['GRU'], ['StackedGRU']], [[32], [64], [128], [32]])

class Exercise1_config(object):
	def __init__(self, cell_type=['LSTM', 'Linear'], hidden_neurons=[32,100], dropout=[None, 0.4], optimizer='Adam', learning_rate=0.001, n_classes=10, n_stack=3):
		super(Exercise1_config, self).__init__()

		self.cell_type = cell_type
		self.hidden_neurons = hidden_neurons
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.n_classes = n_classes
		self.n_stack = n_stack
		self.dropouts = dropout

		self.rnn_cells = list()
		self.weights = list()
		self.biases = list() 

		# self._initialise_weights()
		self.initialise_cells(cell_type)

	def _initialise_weights(self):

		hidden_neurons = self.hidden_neurons
		n_output = self.n_classes

		self.weights.append(tf.Variable(tf.random_normal([hidden_neurons, n_output])))
		self.biases.append(tf.Variable(tf.random_normal([n_output])))

	def initialise_cells(self, cell):

		hidden_neurons = self.hidden_neurons

		for i, n_hidden in enumerate(self.hidden_neurons):

			cell = self.cell_type[i]

			if cell == 'LSTM':
				rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
				if self.dropouts[i] is not None:
					rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(rnn_tf_cell, output_keep_prob=self.dropouts[i])
				self.rnn_cells.append(rnn_tf_cell)
			elif cell == 'GRU':
				rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(n_hidden)
				if self.dropouts[i] is not None:
					rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(rnn_tf_cell, output_keep_prob=self.dropouts[i])
				self.rnn_cells.append(rnn_tf_cell)
				# self.rnn_cells.append(tf.contrib.rnn.core_rnn_cell.GRUCell(n_hidden))
			elif cell == 'StackedLSTM':
				rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
				if self.dropouts[i] is not None:
					rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(rnn_tf_cell, output_keep_prob=self.dropouts[i])
				self.rnn_cells.append(tf.contrib.rnn.core_rnn_cell.MultiRNNCell([rnn_tf_cell for _ in range(self.n_stack)]))
			elif cell == 'StackedGRU':
				rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(n_hidden)
				if self.dropouts[i] is not None:
					rnn_tf_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(rnn_tf_cell, output_keep_prob=self.dropouts[i])
				self.rnn_cells.append(tf.contrib.rnn.core_rnn_cell.MultiRNNCell([rnn_tf_cell for _ in range(self.n_stack)]))
				# self.rnn_cells.append(tf.contrib.rnn.core_rnn_cell.MultiRNNCell([tf.contrib.rnn.core_rnn_cell.GRUCell(n_hidden) for _ in range(self.n_stack)]))
			elif cell == 'Linear':
				self.weights.append(tf.Variable(tf.random_normal([hidden_neurons[i-1], n_hidden])))
				self.biases.append(tf.Variable(tf.random_normal([n_hidden])))

		# output linear transformation
		self.weights.append(tf.Variable(tf.random_normal([n_hidden, self.n_classes])))
		self.biases.append(tf.Variable(tf.random_normal([self.n_classes])))


def define_optimiser(loss, optimizer='GradientDescent', lr=0.001):

	if optimizer=='GradientDescent':
		train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

	elif optimizer=='Adam':
		train_step = tf.train.AdamOptimizer(lr).minimize(loss)

	else:
		raise NotImplementedError('Optimiser not implemented')

	return train_step

def preprocess_data(data, target, max_time=5, exercise=1):

	# binarise values
	data = binarize(data, threshold=0.1)

	# subset values based on max_time (pixel by pixel)
	if max_time is not None:
		data = data[:, 0:max_time]
	else:
		max_time = data.shape[1]

	# reshape to tensor
	if exercise == 1:
		data = data.reshape(-1, max_time, 1)

	elif exercise == 2:
		target =  data[:, 1:] #tf.one_hot(data)
		data = data[:, 0:-1].reshape(-1, max_time-1, 1) # all values except the last

	return data, target

def binarize(images, threshold=0.1):
	return (threshold < images).astype('float32')

def compile_pixel_predictor(x, y_, cells, weights, biases):

	# initialise an LSTM cell
	for i, cell in enumerate(cells):
		# time_major defaults to false here, which is good because our input tensor is [batch_size, max_time, 1]
		n_hidden = int(weights[i].get_shape()[0])
		n_output = int(weights[i].get_shape()[1])
		seq_len = int(x.get_shape()[1])

		init_state = cell.zero_state(FLAGS.batch_size, tf.float32) # sorry world for this global inheritance
		outputs, states = tf.nn.dynamic_rnn(cell, x, initial_state=init_state) 
		rnn_outputs = tf.reshape(outputs, [-1, n_hidden])

		# affine transformation of the cell output
		activations = tf.matmul(rnn_outputs, weights[i]) + biases[i]

		outputs = tf.reshape(activations, [-1, seq_len, n_output])

	return outputs, states, init_state


def compile_lstm(x, cells, weights, biases):

	# initialise an LSTM cell
	for i, cell in enumerate(cells):
		# time_major defaults to false here, which is good because our input tensor is [batch_size, max_time, 1]
		outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) 
		outputs = tf.transpose(outputs, [1, 0, 2])

		# affine transformation of the cell output
		activations = tf.matmul(outputs[-1], weights[i]) + biases[i]

		# always apply relu non-linearity to activations
		activations = tf.nn.relu(activations)

	# final affine transformation for input to the softmax layer
	activations = tf.matmul(activations, weights[-1]) + biases[-1]

	return activations, states

def build_model_graph(model_name, model_config, data_config, exercise=1, train=True):

	# initialise placeholders for x and y
	y_ = None
	if exercise == 2:
		x = tf.placeholder("float", [None, data_config['num_steps']-1, 1], name='input_x')
		# if train:
		loss_weights = tf.ones([FLAGS.batch_size, data_config['num_steps']-1],  name='weights')
		y_ = tf.placeholder(tf.int32, [None, data_config['num_steps']-1], name='target_y') # output 10 values

	elif exercise == 1:
		x = tf.placeholder("float", [None, data_config['num_steps'], 1])
		y_ = tf.placeholder("float", [None, data_config['output_neurons']]) # output 10 values

	# extract model configurations
	optimizer = model_config.optimizer

	weights = model_config.weights
	biases = model_config.biases
	lr = model_config.learning_rate
	cells = model_config.rnn_cells

	# compile the model, evaluate and make predictions
	if exercise == 1:
		y, state = compile_lstm(x, cells, weights, biases)
		init_state = None
		if train:
			loss = categorical_crossentropy(y_, y)

	elif exercise == 2:
		y, state, init_state = compile_pixel_predictor(x, y_,  cells, weights, biases)
		if train:
			loss = tf.contrib.seq2seq.sequence_loss(y, y_, loss_weights)

	# backpropagate using whicever optimiser
	if train:
		train_step = define_optimiser(loss=loss, optimizer=optimizer, lr=lr)
	else:
		train_step = None

	# histograms for the weights and biases
	w_h = tf.summary.histogram("weights", weights[0])
	b_h = tf.summary.histogram("biases", biases[0])

	if train:
		loss_s = tf.summary.scalar("train_loss", loss)

	# now merge
	merged_summary_op = tf.summary.merge_all()

	return train_step, x, y_, y, state, init_state, merged_summary_op

def accuracy(y_, y):
	"""Calculate the accuracy of a set of multi-class predictions"""
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def categorical_crossentropy(y_, y):

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	return cross_entropy

def batch_loss_and_acc(mnist_set, batch_size, exercise=1):
	"""Score up a dataset by batch and then return the average acc and loss"""

	acc = 0
	loss = 0
	num_examples = mnist_set.num_examples
	total_batches = int(num_examples/batch_size)

	for _ in range(total_batches):

		train_xs, train_ys = mnist_set.next_batch(batch_size)
		train_xs, train_ys = preprocess_data(train_xs, train_ys, max_time=data_config['num_steps'], exercise=FLAGS.exercise)
		
		if FLAGS.exercise == 1:
			tmp_acc, tmp_loss = sess.run([acc_eval, mse_eval], feed_dict={x: train_xs, y_: train_ys})
		elif FLAGS.exercise == 2:
			tmp_loss = sess.run([seq_loss], feed_dict={x: train_xs, y_: train_ys})
			tmp_loss = tmp_loss[0]
			tmp_acc = 0

		acc += tmp_acc 
		loss += tmp_loss

	acc /= float(total_batches)
	loss /= float(total_batches)

	return acc, loss

if __name__ == '__main__':

	# check keyword arguments
	if not FLAGS.mnist_path or not FLAGS.save_path:
		print("--mnist_path and --save_path must be specified.")
		sys.exit(1)

	# import dataset with one-hot class encoding
	mnist = input_data.read_data_sets(FLAGS.mnist_path, one_hot=True)

	# define all model architectures requested in the exercises
	if FLAGS.exercise == 1:
		ex_iterator = Q1_HyperParameters
	elif FLAGS.exercise == 2:
		ex_iterator = Q2_HyperParameters

	for cell_types, hidden_units in ex_iterator: # hidden_neurons

		if FLAGS.exercise == 1:
			model_config = Exercise1_config(hidden_neurons=hidden_units, cell_type=cell_types)
		else:
			model_config = Exercise1_config(hidden_neurons=hidden_units, cell_type=cell_types, n_classes=2)

		data_config['num_steps'] = FLAGS.num_steps

		model_name = 'ex' + str(FLAGS.exercise) + '_' + '_'.join(map(str, model_config.cell_type)) + '_' + '_'.join(map(str, model_config.hidden_neurons))

		print('Building model with the {0:s} architecture'.format(model_name))

		# build tensorflow graph
		train_step, x, y_, y, train_state, init_state, summary = build_model_graph(model_name
							, model_config=model_config
							, data_config=data_config
							, exercise=FLAGS.exercise
							)

		# initialise session
		init = tf.global_variables_initializer()
		sess = tf.Session()

		# setup tensorboard
		summary_writer = tf.summary.FileWriter('./tb/', graph_def=sess.graph_def)

		with sess.as_default():
			sess.run(init)
			tf.set_random_seed(SEED)

			# get a mini-batch of validation data
			test_xs, test_ys = mnist.test.next_batch(FLAGS.batch_size)
			test_xs, test_ys = preprocess_data(test_xs, test_ys, max_time=data_config['num_steps'], exercise=FLAGS.exercise)

			# initialise some lists to store performance information
			train_losses = list()
			test_losses = list()
			train_accs = list()
			test_accs = list()
			train_idx = list()
			train_iterations = 0

			# train model
			training_state = None
			for i in range(FLAGS.epochs):

				all_batches = int(mnist.train.num_examples/FLAGS.batch_size)

				for _ in range(all_batches):

					train_iterations += 1

					# train batch and evaluate
					acc_eval = accuracy(y_, y)
					mse_eval = categorical_crossentropy(y_, y)
					batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
					batch_xs, batch_ys = preprocess_data(batch_xs, batch_ys, max_time=data_config['num_steps'], exercise=FLAGS.exercise)

					if FLAGS.exercise == 1:
						_, train_loss, train_acc, summary_str = sess.run([train_step, mse_eval, acc_eval, summary], feed_dict={x: batch_xs, y_: batch_ys })
						test_acc, test_loss = sess.run([acc_eval, mse_eval], feed_dict={x: test_xs, y_: test_ys})
					
					elif FLAGS.exercise == 2:
						seq_loss = tf.contrib.seq2seq.sequence_loss(y, y_, tf.ones([batch_xs.shape[0], FLAGS.num_steps-1]))
						if training_state is not None:
							feed_dict = {x: batch_xs, y_: batch_ys, init_state : training_state}
						else:
							feed_dict = {x: batch_xs, y_: batch_ys}

						_, training_state, train_loss, summary_str = sess.run([train_step, train_state, seq_loss, summary], feed_dict=feed_dict)

						seq_loss = tf.contrib.seq2seq.sequence_loss(y, y_, tf.ones([FLAGS.batch_size, FLAGS.num_steps-1]))
						test_loss = sess.run([seq_loss], feed_dict={x: test_xs, y_: test_ys})
						test_loss = test_loss[0]
						train_acc = 0 
						test_acc = 0

					print('Performance at epoch {0:d}: Accuracy {1:.2f} Loss {2:.2f} Test: Accuracy {3:.2f} Loss {4:.2f}\r' .format(i, train_acc, train_loss, test_acc, test_loss), end="") 
					sys.stdout.flush()

					summary_writer.add_summary(summary_str, train_iterations*FLAGS.batch_size + i)

					# update the performance cache
					log_val = 100
					if train_iterations % log_val == 0: # on every log_val iterations
						train_losses.append(train_loss)
						test_losses.append(test_loss)
						train_accs.append(train_acc)
						test_accs.append(test_acc)
						train_idx.append(train_iterations)


			# calculate total training and test error (over mini-batches, for memory)
			acc, loss = batch_loss_and_acc(mnist.train, FLAGS.batch_size)
			test_acc, test_loss = batch_loss_and_acc(mnist.test, FLAGS.batch_size)

			# plot loss charts
			vis_funcs.plot_train_test_loss(train_idx, train_losses, test_losses, model_name + '_loss')
			vis_funcs.plot_train_test_loss(train_idx, train_accs, test_accs, model_name + '_accuracy')

			# persist tensorflow session
			saver = tf.train.Saver()
			saver.save(sess, './output/' + model_name + '_session.ckpt')

			# determine the final performance of the model and save to disc
			log_funcs.log_model_results('compgi13_2', './logs/', '18/02/17', model_name, model_name, [FLAGS.epochs, FLAGS.batch_size, FLAGS.num_steps], [loss, test_loss, acc, test_acc], './output/' + model_name + '_session.ckpt')

			# clear the graph for next iteration
			tf.reset_default_graph()




