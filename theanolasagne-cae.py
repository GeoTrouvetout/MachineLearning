#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Copyright 2016 geo trouvetout

Created on Wed Sep 28 2016

@author: geo trouvetout
@contact : grj@mailoo.org

"""

import sys, getopt
#import cpickle as pickle

from optparse import OptionParser

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


########### FUNCTIONS DEF ################



"""
simple copy of the function load_dataset() of the lasagne/example/mnist.py
"""
def load_dataset_mnist():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the inputs in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		# The inputs are vectors now, we reshape them to monochrome 2D images,
		# following the shape convention: (examples, channels, rows, columns)
		data = data.reshape(-1, 1, 28, 28)
		# The inputs come as bytes, we convert them to float32 in range [0,1].
		# (Actually to range [0, 255/256], for compatibility to the version
		# provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		return data

	# We can now download and read the training and test set images and labels.
	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]

	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test


def read_model_data(model, filename):
	"""Unpickles and loads parameters into a Lasagne model."""
	filename = os.path.join('./', '%s.%s' % (filename, 'params'))
	with open(filename, 'r') as f:
		data = pickle.load(f)
	lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
	"""Pickels the parameters within a Lasagne model."""
	data = lasagne.layers.get_all_param_values(model)
	filename = os.path.join('./', filename)
	filename = '%s.%s' % (filename, 'params')
	with open(filename, 'w') as f:
		pickle.dump(data, f)

def build_cae(input_var=None):

	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	l_conv1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify )
	l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2) )
	l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify )
	l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2) )
	l_fcenc = lasagne.layers.DenseLayer(l_pool2, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	l_fc = lasagne.layers.DenseLayer(l_fcenc, num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
	l_fcdec = lasagne.layers.DenseLayer(l_fc, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	l_reshpfcdec = lasagne.layers.ReshapeLayer( l_fcdec, shape=(-1, 32, 4, 4) )
	l_upscale1 = lasagne.layers.Upscale2DLayer(l_reshpfcdec, scale_factor=2, mode='repeat')
	l_deconv1 = lasagne.layers.Deconv2DLayer(l_upscale1, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify )
	l_upscale2 = lasagne.layers.Upscale2DLayer(l_deconv1, scale_factor=2, mode='repeat')
	l_deconv2 = lasagne.layers.Deconv2DLayer(l_upscale2, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify )
	l_out = lasagne.layers.FeaturePoolLayer(l_deconv2, 32, pool_function=theano.tensor.max )

	return l_out


"""
simple copy of the function iterate_minibatches(...) of the lasagne/examples/mnist.py
"""
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
		for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
			if shuffle:
				excerpt = indices[start_idx:start_idx + batchsize]
			else:
				excerpt = slice(start_idx, start_idx + batchsize)
				yield inputs[excerpt], targets[excerpt]

############## MAIN ################

def main(num_epochs=10):
	# mnist dataset
	print("Loading mnist data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist()
	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')

	network = build_cae(input_var)

	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.squared_error(prediction, target_var)
	loss_avg = loss.mean()
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss_avg, params, learning_rate=0.1, momentum=0.9)

	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
	test_loss_avg = test_loss.mean()

	test_acc = T.mean( T.eq( T.argmax( test_prediction, axis=1 ), target_var ), dtype=theano.config.floatX )

	train_fn = theano.function([input_var, target_var], loss_avg, updates=updates)

	val_fn = theano.function([input_var, target_var], [test_loss_avg, test_acc])

	print("Starting training...")
	# iteration over epochs:
	for epoch in range(num_epochs):
		train_err = 1
		train_batches = 1
		start_time = time.time()
		for batch in iterate_minibatches(X_train, X_train, 500, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 1
		val_acc = 1
		val_batches = 1
		for batch in iterate_minibatches(X_val, X_val, 500, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1	

		# Then we print the results for this epoch:
		print( "Epoch {} of {} took {:.3f}s".format( epoch + 1, num_epochs, time.time() - start_time) )
		print( "	training loss:\t\t{:.6f}".format(train_err / train_batches) )
		print( "	validation loss:\t\t{:.6f}".format(val_err / val_batches) )
		print( "	validation accuracy:\t\t{:.2f} %".format( val_acc / val_batches * 100) )

	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Final results:")
	print("	test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("	test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))


if __name__ == "__main__":
	main()
