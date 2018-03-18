"""
LSTM TensorFlow Assignments
Advanced Topics in Machine Learning, UCL (COMPGI13)
Coursework 2
Run LSTM image generators (e.g. in-painting)
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
import joblib

# tensorflow data processing functions
from tensorflow.examples.tutorials.mnist import input_data

# joblib for dumping and loading model classes
import joblib

# matplotlib for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# own functions
import vis_funcs
import log_funcs
from run_lstm import Exercise1_config, data_config, build_model_graph

# random seed
SEED = 42
np.random.seed(SEED)

# run config
flags = tf.app.flags

# FIXME: create separate run files so that other flags aren't inherited
flags.DEFINE_string("mnist_path_", '../Coursework_1/input_data/', 'Directory of mnist input')
flags.DEFINE_string("save_path_", '../output/', "Directory to write the model.")
flags.DEFINE_integer("exercise_", 3, "What exercise would you like to generate sequences for?")
flags.DEFINE_integer("generate", 1, "Do you want to generate sequences for the 100 samples (1=Yes)?")

FLAGS = flags.FLAGS

# model config
OUTPUT_PATH = './output/'
EXERCISE_2_MODELS = {

		'GRU_32' : {'file' : 'ex2_GRU_32_session.ckpt', 'cells' : ['GRU'], 'neurons' : [32]}
	,	'GRU_64' : {'file' : 'ex2_GRU_64_session.ckpt', 'cells' : ['GRU'], 'neurons' : [64]}
	, 	'GRU_128': {'file' : 'ex2_GRU_128_session.ckpt', 'cells' : ['GRU'], 'neurons' : [128]}
	,	'StackedGRU_32' : {'file' : 'ex2_StackedGRU_32_session.ckpt', 'cells' : ['StackedGRU'], 'neurons' : [32]}

}

def binarize(images, threshold=0.1):
	return (threshold < images).astype('float32')

def all_activations(repeat=2):
	possible_activations = list()
	for i in itertools.product([0, 1], repeat=repeat):
		possible_activations.append(i)
	return possible_activations

def generate_ex3_dataset(dataset, possible_activations):
	expanded_dset = list()

	for val in possible_activations:
		dset = copy.deepcopy(dataset)
		for row in range(dset.shape[0]):
			dset[row, :][dset[row, :] == -1] = val

		expanded_dset.append(dset)
		
	return np.vstack(expanded_dset)

def load_ex3_data(pixels=1):
	# load in datasets
	if pixels == 1:
		dataset = np.load('./inpainting_data/one_pixel_inpainting.npy')

	elif pixels == 2:
		dataset = np.load('./inpainting_data/2X2_pixels_inpainting.npy')
	
	images    = dataset[0]
	gt_images = dataset[1] 

	return images, gt_images

# def generate_pixel_combinations(pixels=1):

# 	possible_activations = list()
# 	if pixels == 1:
# 		for i in itertools.product([0, 1], repeat=2):
# 			possible_activations.append(i)

# 	if pixels == 2:
# 		for i in itertools.product([0, 1], repeat=4):
# 			possible_activations.append(i)

# 	return possible_activations

def ex3_mnist_inpainting(mnist_path, model_config, data_config, model_checkpoint):
	"""
	PLAN:
	
	For all one-pixel in-paintings:
		For all 2x2 pixel in-paintings:
			For each GRU model trained in exercise 2:
				For each possibility of the missing pixels:
					Compute a forward pass of the GRU for all pixels, calculating model probabilities for each input
						Calculate the cross entropy over that whole image (i.e. compute most likely image according to the model)
						Save the value with the best xentropy

	"""
	gt_xents = list()
	gt_xents_tmp = list() # revisit this
	final_xents = dict()
	final_images = dict()

	for pixel in [1, 2]: # for 1x1 and 2x2 pixel combos

		images, gt_images = load_ex3_data(pixels=pixel)
		data_config['num_steps'] = 2 

		total_pixels = images.shape[1] 

		# generate all possible sequences of missing pixels
		print('creating dset')
		if pixel == 1:
			poss_activations = all_activations(repeat=1)
			images = generate_ex3_dataset(images, poss_activations)

		elif pixel == 2:
			poss_activations = all_activations(repeat=4)
			images = generate_ex3_dataset(images, poss_activations)

		print('compiling TF graph')
		train_step, x, y_, y, state, init_state, _ = build_model_graph('', model_config=model_config
					, data_config=data_config, exercise=2, train=False)

		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, model_checkpoint)

			# now make predictions 
			cell_state = None
			for i in range(total_pixels-1): # all pixels but the last
				batch_start = 0
				for batch in range(int(images.shape[0] / FLAGS.batch_size)):

					# current pixel is x and next pixel is y
					pixel_x = images[batch_start:(batch_start+FLAGS.batch_size), i]
					target_pixel = images[batch_start:(batch_start+FLAGS.batch_size), i+1].reshape([-1, 1])
					pixel_x = pixel_x.reshape([-1,1,1])

					# generate predictions and sequence loss for that pixel
					seq_loss = tf.contrib.seq2seq.sequence_loss(y, y_, tf.ones([FLAGS.batch_size, 1]), average_across_batch=False)

					if cell_state is not None:
						in_dict = {x: pixel_x, y_ : target_pixel, init_state : cell_state}
					else:
						in_dict = {x: pixel_x, y_ : target_pixel}

					# make predictions and xent loss for ground truth
					preds, cell_state, train_loss = sess.run([y, state, seq_loss], feed_dict=in_dict)
					batch_start += FLAGS.batch_size

					# we go in batches of 200
					# over each of the 783 pixels
					# for each pixel dataset

					# so for each batch, we want to row-append our losses (ending up with as many losses as rows in our dset)
					# then for each pixel we want to column stack, so that we then have 2000 x 1, 2000 x 2... 2000 x 784

					gt_xents.append(train_loss) #FIX this needs to be interpretable

				gt_xents_tmp.append(np.hstack(gt_xents))
				gt_xents = list()
			
		final_xents[pixel] = np.vstack(gt_xents_tmp).T
		final_images[pixel] = images
		gt_xents_tmp = list()
		tf.reset_default_graph()

	return final_xents, final_images

def group(lst, n):
	"""group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]
	
	Group a list into consecutive n-tuples. Incomplete tuples are
	discarded e.g.
	
	>>> group(range(10), 3)
	[(0, 1, 2), (3, 4, 5), (6, 7, 8)]
	"""
	return zip(*[lst[i::n] for i in range(n)]) 

def process_x3_entropies(final_xents, images):

	"""
	if pixel = 2 then 2x2 pixels meaning there are 16 possibilities for every image
	If pixel = 1 then 1x1 pixels. Meaning there are two possibilities for every image, so we:

	1. take row-wise mean of xentropies
	2. pick lowest xentropy for each 2/16 rows
	3. return the image with the lowest xentropy

	"""

	best_images = dict()

	for key in final_xents:

		xents = final_xents[key].mean(axis=1)
		assert(xents.shape[0] == final_xents[key].shape[0])

		# find the minimum of group of rows so as to find the lowest xentropy image
		if key == 1:
			groups = 2
		elif key == 2:
			groups = 16
		
		best_idx = [x[np.argmin(images_mean[list(x)])] for x in group(range(len(images_mean)), groups)]
		best_images[key] = images[best_idx, :] # images with the lowest xentropy according to model

	return best_images

def ex2_mnist_inpainting(mnist_path, model_config, data_config, model_checkpoint, mnist_samples=100, remove_pixels=300):

	FLAGS.batch_size = mnist_samples

	# sample mnist_samples from mnist test set
	mnist = input_data.read_data_sets(mnist_path, one_hot=True)
	test_xs, _ = mnist.test.next_batch(mnist_samples)
	test_xs = binarize(test_xs, threshold=0.1)

	total_pixels = test_xs.shape[1] 

	data_config['num_steps'] = 2 # bit hacky, but note that we subtract 1 from num_steps within the build graph function

	train_step, x, y_, y, state, init_state, _ = build_model_graph('', model_config=model_config
						, data_config=data_config, exercise=2, train=False)

	# load in models
	generation = False
	gt_xents = list()
	sim_xents = list()
	sim_pixels = test_xs

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, model_checkpoint)

		# now make predictions 
		cell_state = None
		for i in range(total_pixels-1): # all pixels but the last

			# current pixel is x and next pixel is y
			pixel = test_xs[:, i]
			target_pixel = test_xs[:, i+1].reshape([-1, 1])

			if i+1 == (total_pixels - remove_pixels): # if we're in the removed part of the image
				orig_pixel = pixel.reshape([-1,1,1])
				pixel = pred_pixel.eval()
				generation = True

				# if we're generating, calculate predictions for the real x and the generated x
				#
			
			pixel = pixel.reshape([-1,1,1])

			# generate predictions and sequence loss for that pixel
			seq_loss = tf.contrib.seq2seq.sequence_loss(y, y_, tf.ones([test_xs.shape[0], 1]))

			if cell_state is not None:
				in_dict = {x: pixel, y_ : target_pixel, init_state : cell_state}
			else:
				in_dict = {x: pixel, y_ : target_pixel}

			# make predictions and xent loss for ground truth
			preds, cell_state, train_loss = sess.run([y, state, seq_loss], feed_dict=in_dict)
			preds = tf.nn.sigmoid(preds)
			pred_pixel = tf.argmax(preds, axis=2)

			if generation:
				in_dict = {x: orig_pixel, y_ : target_pixel}
				gt_loss = sess.run(seq_loss, feed_dict=in_dict)

			# calculate xent loss of generated image
			# smp_seq_loss = tf.contrib.seq2seq.sequence_loss(y, pred_pixel, tf.ones([test_xs.shape[0], 1]))
			# smp_loss = sess.run(smp_seq_loss, feed_dict=in_dict)

			if generation:
				gt_xents.append(gt_loss)
				sim_xents.append(train_loss)
				sim_pixels[:, i+1] = pred_pixel.eval().reshape(-1)

	return sim_pixels, gt_xents, sim_xents


def summarise_ex2_losses(gt_losses, sim_losses, seq_length=[1, 10, 28, 300]):

	return_dict = dict()

	for seq in seq_length:

		if seq_length > 1:
			gt_loss = np.mean(gt_losses[::-1][0:seq])
			sim_loss = np.mean(sim_losses[::-1][0:seq])
		else:
			gt_loss = gt_losses[0]
			sim_loss = sim_losses[0]

		return_dict[seq] = {'ground_truth' : gt_loss, 'simulated' : sim_loss}

	return return_dict

def plot_mnist(pixels, title=None, save_path=None, denote_gen_start=True):

	# reshape to a matrix
	pixels = pixels.reshape([28, 28])

	# plot mnist digit
	fig,ax = plt.subplots(1)
	ax.imshow(pixels, cmap='gray')
	if title is not None:
		plt.title(title)

	plt.show()

	# add rectangle to start point of sequence
	if denote_gen_start:
		y = (784 - 300)
		x = 28 * int((y / 28.) - int(y / 28))
		y = int(y / 28)
		rect = patches.Rectangle((x,y),1,1,linewidth=1,edgecolor='r',facecolor='none')

		# Add the patch to the Axes
		ax.add_patch(rect)

	if save_path is not None:
		plt.savefig(save_path, bbox_inches='tight')

def select_images_to_plot(images, targets, number=5, save_path='./'):

	random_image_ixs = np.random.choice(images.shape[0], number)

	for ix in random_image_ixs:
		plot_mnist(images[ix, :], title='Image {0:d} (Target={1:d})'.format(ix, targets[ix])
			, save_path=save_path + '{0:d}_gen_mnist.png'.format(ix), denote_gen_start=True)

def init_animation():
	pixi = copy.deepcopy(pixels)
	pixi[:, n_start:] = 0
	cnt = 0
	for im_ix, x in enumerate(ax):
		x.imshow(pixi[im_ix].reshape([28,28]), cmap='gray')
		x.set_title('Target = {0:d}'.format(target_y[cnt]))
		x.get_xaxis().set_visible(False)
		x.get_yaxis().set_visible(False)
		cnt += 1

def animate(i):
	pix = copy.deepcopy(pixels)
	pix[:, (n_start) + (i * jump):] = 0
	for im_ix, x in enumerate(ax):
		x.imshow(pix[im_ix].reshape([28,28]), cmap='gray')

def generate_gif(image, target, start=784-280):
	global pixels, ax, n_start, jump, target_y 

	n_start = start
	pixels = image
	jump = 10 # jump this many pixels forward on each iterate
	target_y = target

	# initialise plot
	fig, ax = plt.subplots(1, image.shape[0], figsize=(4,2))

	# generate animation
	ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=30)
	return ani

if __name__ == '__main__':

	if FLAGS.exercise_ == 2:

		for model_name in EXERCISE_2_MODELS.keys():

			tf.reset_default_graph()

			ckpt_path = OUTPUT_PATH + EXERCISE_2_MODELS[model_name]['file']
			hidden_neurons = EXERCISE_2_MODELS[model_name]['neurons']
			cell_types = EXERCISE_2_MODELS[model_name]['cells']

			model_config = Exercise1_config(hidden_neurons=hidden_neurons, cell_type=cell_types, n_classes=2)

			if FLAGS.generate == 1:
				# generate sequences a la exercise 2
				sim_images, gt_losses, sim_losses = ex2_mnist_inpainting(FLAGS.mnist_path_, model_config, data_config, ckpt_path, mnist_samples=100, remove_pixels=300)

				# dump the images to a file
				joblib.dump(sim_images, FLAGS.save_path_ + model_name + '_simulated_images.pkl')

				# calculate losses from exercise 2b, namely:
				# xent losses for the next 1, 10, 28, 300 pixels
				xent_losses = summarise_ex2_losses(gt_losses, sim_losses, seq_length=[1, 10, 28, 300])
				print('The xentropy losses for {1:s} requested in Exercise 2B are: {0:s}' .format(xent_losses, model_name))

			else:
				# load in the simulated images
				sim_images = joblib.load(FLAGS.save_path_ + model_name + '_simulated_images.pkl')

			# load in the target values
			mnist = input_data.read_data_sets(FLAGS.mnist_path_, one_hot=True)
			_, test_ys = mnist.test.next_batch(100)

			# # generate sample images
			# select_images_to_plot(sim_images, np.argmax(test_ys, axis=1), number=6, save_path=FLAGS.save_path_)

			# plot image completions for a successful, failure and high-variance sample
			images_to_plot = (34,51,38) # note: samples found outside of this code
			ani = generate_gif(sim_images[images_to_plot, :], target=np.argmax(test_ys[images_to_plot,:], axis=1), start=784-280)
			ani.save('./{0:s}_animation.gif'.format(model_name), writer='imagemagick', fps=20)

	if FLAGS.exercise_ == 3: # pixel in-painting using future values

		# for model_name in EXERCISE_2_MODELS.keys():

		model_name = 'GRU_128'
		tf.reset_default_graph()

		ckpt_path = OUTPUT_PATH + EXERCISE_2_MODELS[model_name]['file']
		hidden_neurons = EXERCISE_2_MODELS[model_name]['neurons']
		cell_types = EXERCISE_2_MODELS[model_name]['cells']

		model_config = Exercise1_config(hidden_neurons=hidden_neurons, cell_type=cell_types, n_classes=2)

		# generate sequences a la exercise 2
		xents, exp_images = ex3_mnist_inpainting(FLAGS.mnist_path_, model_config, data_config, ckpt_path)
		best_images = process_x3_entropies(xents, exp_images)

		joblib.dump(best_images, './output/ex3_probable_images.pkl')



