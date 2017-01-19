#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Copyright 2017 GeoTrouvetout

Created on Jan 2017

@author: geo trouvetout
@contact : grj@mailoo.org
@github : https://github.com/GeoTrouvetout/

"""

import sys, getopt
#import cpickle as pickle

import argparse

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import nolearn


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


def read_model_data(filename):
	model = np.load(filename)
	return model

def write_model_data(model, filename):
	"""Pickels the parameters within a Lasagne model."""
	data = lasagne.layers.get_all_param_values(model)
	filename = os.path.join('./', filename)
	filename = '%s.%s' % (filename, 'npz')
	np.savez(filename, *lasagne.layers.get_all_param_values(model))

def build_gcae(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	l_conv1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2) )
	l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2) )
	l_fcenc = lasagne.layers.DenseLayer(l_pool2, num_units=512, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	l_fc = lasagne.layers.DenseLayer(l_fcenc, num_units=256, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	
	l_fcdec = lasagne.layers.DenseLayer(l_fc, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	l_reshpfcdec = lasagne.layers.ReshapeLayer( l_fcdec, shape=(-1, 32, 4, 4) )
	l_upscale1 = lasagne.layers.Upscale2DLayer(l_reshpfcdec, scale_factor=2, mode='repeat')
	l_deconv1 = lasagne.layers.Deconv2DLayer(l_upscale1, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	l_upscale2 = lasagne.layers.Upscale2DLayer(l_deconv1, scale_factor=2, mode='repeat')
	l_deconv2 = lasagne.layers.Deconv2DLayer(l_upscale2, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	l_out = lasagne.layers.FeaturePoolLayer(l_deconv2, 32, pool_function=theano.tensor.max )
	
	l_outclass = lasagne.layers.DenseLayer(l_fc, num_units=10, nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.GlorotUniform())
	
	
	print(lasagne.layers.get_output_shape(l_out))
	return l_out, l_outclass



def build_lae(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	print(lasagne.layers.get_output_shape(l_in))

	network = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2) )
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Conv2DLayer( network, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.MaxPool2DLayer( network, pool_size=(2,2) )
	print(lasagne.layers.get_output_shape(network))
	
# 	l_fcenc = lasagne.layers.DenseLayer(l_pool2, num_units=512, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
# 	l_fc = lasagne.layers.DenseLayer(l_fcenc, num_units=256, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	l_le = lasagne.layers.Conv2DLayer( network, num_filters=10, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	print(lasagne.layers.get_output_shape(l_le))
	
	network = lasagne.layers.Conv2DLayer( l_le, num_filters=32, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	print(lasagne.layers.get_output_shape(network))
	
# 	l_fcdec = lasagne.layers.DenseLayer(l_fc, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
# 	l_reshpfcdec = lasagne.layers.ReshapeLayer( l_fcdec, shape=(-1, 32, 4, 4) )
# 	l_upscale1 = lasagne.layers.Upscale2DLayer(l_reshpfcdec, scale_factor=2, mode='repeat')

	network = lasagne.layers.Upscale2DLayer(network, scale_factor=2, mode='repeat')
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Deconv2DLayer( network, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Upscale2DLayer(network, scale_factor=2, mode='repeat')
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Deconv2DLayer(network, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	print(lasagne.layers.get_output_shape(network))
	
	l_out = lasagne.layers.FeaturePoolLayer(network, 32, pool_function=theano.tensor.max )
	
	l_outclass = lasagne.layers.DenseLayer(l_le, num_units=10, nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.GlorotUniform())
	
	
	print(lasagne.layers.get_output_shape(l_out))
	return l_out, l_outclass



"""
simple copy of the function iterate_minibatches(...) of the lasagne/examples/mnist.pyo
"""

def iterate_minibatches(inputs, targets, classes, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt], classes[excerpt]


############## MAIN ################

def main(num_epochs=10):
	# mnist dataset
	print("Loading mnist data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist()
	X_train_Out = np.copy(X_train)
	print(len(X_train))
	
	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')
	class_var = T.ivector('classes')
	
	print("Loading mnist data...")
	network_enc, network_class = build_lae(input_var)
	
	prediction_enc = lasagne.layers.get_output(network_enc)
	loss_enc = lasagne.objectives.squared_error(prediction_enc, target_var)
	aggregated_loss_enc = lasagne.objectives.aggregate(loss_enc)
	params_enc = lasagne.layers.get_all_params(network_enc, trainable=True)
	updates_enc = lasagne.updates.nesterov_momentum(aggregated_loss_enc, params_enc, learning_rate=0.1, momentum=0.9)
	
	train_fn_enc = theano.function([input_var, target_var], aggregated_loss_enc, updates=updates_enc)
	
	test_prediction_enc = lasagne.layers.get_output(network_enc, deterministic=True)
	
	test_loss_enc = lasagne.objectives.squared_error(test_prediction_enc, target_var)
	aggregated_test_loss_enc = lasagne.objectives.aggregate(test_loss_enc)
	
	prediction_class = lasagne.layers.get_output(network_class)
	loss_class = lasagne.objectives.categorical_crossentropy(prediction_class, class_var)
	aggregated_loss_class = lasagne.objectives.aggregate(loss_class)
	params_class = lasagne.layers.get_all_params(network_class, trainable=True)
	updates_class = lasagne.updates.nesterov_momentum(aggregated_loss_class, params_class, learning_rate=0.1, momentum=0.9)
	
	train_fn_class = theano.function([input_var, class_var], aggregated_loss_class, updates=updates_class)
	
	test_prediction_class = lasagne.layers.get_output(network_class, deterministic=True)
	test_loss_class = lasagne.objectives.categorical_crossentropy(test_prediction_class, class_var)
	test_acc_class = T.mean( T.eq( T.argmax( test_prediction_class, axis=1 ), class_var ), dtype=theano.config.floatX )
	aggregated_test_loss_class = lasagne.objectives.aggregate(test_loss_class)
	
	
	train_fn_glob = theano.function([input_var, class_var, target_var], aggregated_loss_class+aggregated_loss_enc, updates=updates_class)
	
	val_fn_class = theano.function([input_var, class_var], aggregated_test_loss_class)
	val_fn_enc = theano.function([input_var, target_var], aggregated_test_loss_enc)
	val_fn_glob = theano.function([input_var, class_var, target_var], [aggregated_loss_class+aggregated_loss_enc, test_acc_class] )
	
	
	print("Starting training...")
	# iteration over epochs:
	overall_time = time.time()
	for epoch in range(num_epochs):
		train_err = 0
		train_acc = 0
		train_err_enc = 0
		train_err_class = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, X_train, y_train, 100, shuffle=True):
			inputs, targets, classes = batch
			# train_err_enc += train_fn_enc(inputs, targets)
			# train_err_class += train_fn_class(inputs, classes)
			train_err += train_fn_glob(inputs, classes, targets)
			err, acc= val_fn_glob(inputs, classes, targets)
			train_acc += acc
			train_batches += 1
			print(" minibatch",train_batches, "of epoch", epoch + 1, ":", "\t{:.3f}s".format( time.time() - start_time) )
			print("\t training loss:\t\t{:.6f} ".format(train_err / train_batches) )
			print("\t training accuracy:\t{:.2f} %".format( 100*(train_acc / train_batches) ) )		
	
			# print("		training loss enc =", train_err_enc / train_batches) 
			# print("		training loss class =", train_err_class / train_batches) 
	
		print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
		print("\t training loss:\t\t{:.6f} ".format(train_err / train_batches) )
		print("\t training accuracy:\t{:.2f} %".format( 100*(train_acc / train_batches) ) )		
	
		# print("		training loss enc =", train_err_enc / train_batches) 
		# print("		training loss class =", train_err_class / train_batches) 
		# And a full pass over the validation data:
		val_err = 0
		val_err_enc = 0
		val_err_class = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, X_val, y_val, 100, shuffle=True):
			inputs, targets, classes = batch
			val_err_enc += val_fn_enc(inputs, targets)
			val_err_class += val_fn_class(inputs, classes)
			err, acc= val_fn_glob(inputs, classes, targets)
			val_err += err
			val_acc += acc
			val_batches += 1
		# print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
		print("\t validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("\t validation encoder loss:\t{:.6f}".format(val_err_enc / val_batches) )
		print("\t validation class loss:\t\t{:.6f}".format(val_err_class / val_batches) )
		print("\t validation accuracy:\t\t{:.2f} %".format( 100*(val_acc / val_batches) ) )		
		# print("		training loss class =", train_err_class / train_batches) 
		# Then we print the results for this epoch:
		# print( "Epoch {} of {} took {:.3f}s".format( epoch + 1, num_epochs, time.time() - start_time) )
		# print( "	training loss:\t\t{:.6f}".format(train_err / train_batches) )
		# print( "	validation loss:\t\t{:.6f}".format(val_err / val_batches) )
		# print( "	validation accuracy:\t\t{:.2f} %".format( val_acc / val_batches * 100) )
	
	test_err = 0
	test_err_enc = 0
	test_err_class = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, X_test, y_test, 500, shuffle=True):
		inputs, targets, classes = batch
		test_err_enc += val_fn_enc(inputs, targets)
		test_err_class += val_fn_class(inputs, classes)
		err, acc= val_fn_glob(inputs, classes, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
	print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
	print("\t test loss:\t\t{:.6f}".format(test_err / test_batches))
	print("\t test encoder loss:\t{:.6f}".format(test_err_enc / test_batches) )
	print("\t test class loss:\t{:.6f}".format(test_err_class / test_batches) )
	print("\t test accuracy:\t\t{:.2f} %".format( 100*(test_acc / test_batches) ) )		
	
	write_model_data(network_enc, 'network_enc')
	write_model_data(network_class, 'network_class')
	

if __name__ == "__main__":
	print("guided fully-connected convolutional autoencoder")
	main()
