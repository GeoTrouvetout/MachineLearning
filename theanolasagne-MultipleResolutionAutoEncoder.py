#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Copyright 2016 geo trouvetout

Created on Mon Jan 09 2017

@author: geo trouvetout
@contact : grj@mailoo.org

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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def transform_dataset_mnist(X_train, y_train, X_val, y_val, X_test, y_test, newsize=280):
	X_train = np.append( X_train, np.zeros( ( 2*len(X_train), 1, 28,28 ) ).astype(int), axis=0 )
	y_train = np.append( y_train, np.zeros( ( 2*len(y_train) ) ).astype(int), axis=0 )
	ltrain = len(X_train)
	indtrain = np.arange( ltrain )
	np.random.shuffle(indtrain)
	X_train = X_train[indtrain]
	X_train = X_train.reshape( -1 , 1, newsize, 28 )
	X_train = np.swapaxes(X_train, 2,3)
	X_train = X_train.reshape( -1 , 1, newsize, newsize )
	X_train = np.swapaxes(X_train, 2,3)
	y_train = y_train[indtrain]

	X_val = np.append( X_val, np.zeros( ( 2*len(X_val), 1, 28,28 ) ).astype(int), axis=0 )
	y_val = np.append( y_val, np.zeros( ( 2*len(y_val) ) ).astype(int), axis=0 )
	lval = len(X_val)
	indval = np.arange( lval )
	np.random.shuffle(indval)
	X_val = X_val[indval]
	X_val = X_val.reshape( -1 , 1, newsize, 28 )
	X_val = np.swapaxes(X_val, 2,3)
	X_val = X_val.reshape( -1 , 1, newsize, newsize )
	X_val = np.swapaxes(X_val, 2,3)
	y_val = y_val[indval]

	X_test = np.append( X_test, np.zeros( ( 2*len(X_test), 1, 28,28 ) ).astype(int), axis=0 )
	y_test = np.append( y_test, np.zeros( ( 2*len(y_test) ) ).astype(int), axis=0 )
	ltest = len(X_test)
	indtest = np.arange( ltest )
	np.random.shuffle(indtest)
	X_test = X_test[indtest]
	X_test = X_test.reshape( -1 , 1, newsize, 28 )
	X_test = np.swapaxes(X_test, 2,3)
	X_test = X_test.reshape( -1 , 1, newsize, newsize )
	X_test = np.swapaxes(X_test, 2,3)
	y_test = y_test[indtest]

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

def build_cae(input_var=None):

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
	print(lasagne.layers.get_output_shape(l_out))
	return l_out


def build_lae(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	l_conv = lasagne.layers.Conv2DLayer(l_in, num_filters=5, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	# l_enc = lasagne.layers.Conv2DLayer(l_conv, num_filters=10, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	l_deconv = lasagne.layers.Deconv2DLayer(l_conv, 5, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	l_out = lasagne.layers.FeaturePoolLayer(l_deconv, 5, pool_function=theano.tensor.max )
	print("network built : lae : ")
	print("		", lasagne.layers.get_output_shape(l_in))
	print("		", lasagne.layers.get_output_shape(l_conv))
	# print("		", lasagne.layers.get_output_shape(l_enc))
	print("		", lasagne.layers.get_output_shape(l_deconv))
	# print("		", lasagne.layers.get_output_shape(l_out))
	print("		", lasagne.layers.get_output_shape(l_out))
	return l_out


def build_mrae(input_var=None):
	
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 280, 280), input_var=input_var)

	l_c1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	l_c1_d = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2,2) )

	l_c2 = lasagne.layers.Conv2DLayer(l_c1_d, num_filters=32, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	l_c2_d = lasagne.layers.MaxPool2DLayer(l_c2, pool_size=(2,2) )

	l_c3 = lasagne.layers.Conv2DLayer(l_c2_d, num_filters=32, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	# l_c3_d = lasagne.layers.MaxPool2DLayer(l_c3, pool_size=(2,2) )

	# network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2) )

	l_fe = lasagne.layers.Conv2DLayer(l_c3, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )

	# network = lasagne.layers.DenseLayer(network, num_units=512, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	# network = lasagne.layers.DenseLayer(network, num_units=256, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	# network = lasagne.layers.DenseLayer(network, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	
	# network = lasagne.layers.ReshapeLayer( network, shape=(-1, 32, 4, 4) )
	# l_d0_3_u = lasagne.layers.Upscale2DLayer(l_fe, scale_factor=2, mode='repeat')
	l_d0_3 = lasagne.layers.TransposedConv2DLayer(l_fe, 32, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())

	l_d0_2_u = lasagne.layers.Upscale2DLayer(l_d0_3, scale_factor=2, mode='repeat')
	l_d0_2 = lasagne.layers.TransposedConv2DLayer(l_d0_2_u, 32, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())

	l_d0_1_u = lasagne.layers.Upscale2DLayer(l_d0_2, scale_factor=2, mode='repeat')
	l_d0_1 = lasagne.layers.TransposedConv2DLayer(l_d0_1_u, 32, filter_size=(3,3), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())

	l_out = lasagne.layers.FeaturePoolLayer(l_d0_1, 32, pool_function=theano.tensor.max )
	
	print("Shape of the created network :")
	print("\t", lasagne.layers.get_output_shape(l_in))
	print("\t", lasagne.layers.get_output_shape(l_c1))
	print("\t", lasagne.layers.get_output_shape(l_c1_d))
	print("\t", lasagne.layers.get_output_shape(l_c2))
	print("\t", lasagne.layers.get_output_shape(l_c2_d))
	print("\t", lasagne.layers.get_output_shape(l_c3))
	# print("\t", lasagne.layers.get_output_shape(l_c3_d))
	print("\t", lasagne.layers.get_output_shape(l_fe))
	# print("\t", lasagne.layers.get_output_shape(l_d0_3_u))
	print("\t", lasagne.layers.get_output_shape(l_d0_3))
	print("\t", lasagne.layers.get_output_shape(l_d0_2_u))
	print("\t", lasagne.layers.get_output_shape(l_d0_2))
	print("\t", lasagne.layers.get_output_shape(l_d0_1_u))
	print("\t", lasagne.layers.get_output_shape(l_d0_1))
	print("\t", lasagne.layers.get_output_shape(l_out))
	print("\t Choices have been made... ")
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
	X_train, y_train, X_val, y_val, X_test, y_test = transform_dataset_mnist(X_train, y_train, X_val, y_val, X_test, y_test, 280 )

	print(len(X_train))

	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')

	network = build_mrae(input_var)

	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.squared_error(prediction, target_var)
	# loss_avg = loss.mean()
	aggregated_loss = lasagne.objectives.aggregate(loss)


	params = lasagne.layers.get_all_params(network, trainable=True)

	updates = lasagne.updates.nesterov_momentum(aggregated_loss, params, learning_rate=0.1, momentum=0.9)

	train_fn = theano.function([input_var, target_var], aggregated_loss, updates=updates)


	#test_acc = T.mean( T.eq( T.argmax( test_prediction, axis=1 ), target_var ), dtype=theano.config.floatX )

	# train_fn = theano.function([input_var, target_var], loss_avg, updates=updates)

	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	predict_fn = theano.function([input_var], test_prediction)

	test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
	# test_loss_avg = test_loss.mean()
	aggregated_test_loss = lasagne.objectives.aggregate(test_loss)


	val_fn = theano.function([input_var, target_var], aggregated_test_loss)

	print("Starting training...")
	overall_time = time.time()

	# iteration over epochs:
	for epoch in range(num_epochs):
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, X_train, 50, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1
			print(" minibatch",train_batches, " ( epoch", epoch + 1, ") :", "\t{:.3f}s".format( time.time() - start_time) )
			print("\t training loss:\t\t{:.6f} ".format(train_err / train_batches) )

		print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
		print("\t training loss:\t\t{:.6f} ".format(train_err / train_batches) )
		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, X_val, 50, shuffle=False):
			inputs, targets = batch
			err = val_fn(inputs, targets)
			val_err += err
			val_batches += 1
		print("\t validation loss:\t\t{:.6f}".format(val_err / val_batches))
		# Then we print the results for this epoch:
		# print( "Epoch {} of {} took {:.3f}s".format( epoch + 1, num_epochs, time.time() - start_time) )
		# print( "	training loss:\t\t{:.6f}".format(train_err / train_batches) )
		# print( "	validation loss:\t\t{:.6f}".format(val_err / val_batches) )
		# print( "	validation accuracy:\t\t{:.2f} %".format( val_acc / val_batches * 100) )

	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, X_test, 50, shuffle=False):
		inputs, targets = batch
		err = val_fn(inputs, targets)
		test_err += err
		# test_acc += acc
		test_batches += 1
	print("Final results:")
	print("\t test loss:\t\t{:.6f}".format(test_err / test_batches))
	print("Total time :\t{:.3f}s".format( time.time() - overall_time))

	# print("	test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
	write_model_data(network, 'network')


	# print("Display some results: ")
	# indices = np.arange(len(X_test))
	# np.random.shuffle(indices)
	# X_test_shuffled = X_test[indices]

	# nbsample = 10
	# X_sample = X_test_shuffled[ 0:nbsample,:,:,:]
	# #displayImage(X_sample[0])

	# result = predict_fn(X_sample)
	# print("X_sample.shape",X_sample.shape)
	# print("result.shape",result.shape)

	# for i in range(len(X_sample)):
	# 	vis1 = np.hstack([X_sample[i*nbsample+j,0,:,:] for j in range(nbsample)])
	# 	vis2 = np.hstack([result[i*nbsample+j,0,:,:] for j in range(nbsample)])
	# 	plt.imshow(np.vstack([vis1,vis2]))
	# 	plt.show()
	


if __name__ == "__main__":
	main()
