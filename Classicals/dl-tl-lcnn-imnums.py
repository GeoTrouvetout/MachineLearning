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

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import stats

import theano
import theano.tensor as T

import lasagne
import nolearn


########### FUNCTIONS DEF ################

def displayImage(image):
	plt.imshow(image, cmap = plt.get_cmap('gray'))
	plt.show()

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
		return (data / np.float32(256))
# 		return 1-(2*(data / np.float32(256)))

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
# 	X_train, X_val = X_train[:-10000], X_train[-10000:]
# 	y_train, y_val = y_train[:-10000], y_train[-10000:]
	
	
	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_test, y_test

def split_data(X, Y, proportion=20):
	T_ind = np.arange(len(Y))
	np.random.shuffle(T_ind)
	X = X[T_ind]
	Y = Y[T_ind]
	
	nb_p = np.floor( ( proportion/100 ) * len(X) ).astype(int)  # part used for the supervised learning
	nb_ip = np.floor( (1-(proportion/100)) * len(X) ).astype(int)  # part used for the autoencoder (the rest for training the classifier)
	if nb_p !=0:
		# supervised / non-supervised split
		X_ip, X_p = X[:-nb_p], X[-nb_p:]
		Y_ip, Y_p = Y[:-nb_p], Y[-nb_p:]
	elif nb_p == 0: # if supervision Rate 100% -> p = 0 -> the selection of indices [-0:] and [:-0] are permuted
		X_p, X_ip = X[:-nb_p], X[-nb_p:]
		Y_p, Y_ip = Y[:-nb_p], Y[-nb_p:]

	return X_p, Y_p, X_ip, Y_ip


def read_model_data(filename):
	model = np.load(filename)
	return model

def write_model_data(model, filename):
	"""Pickels the parameters within a Lasagne model."""
	data = lasagne.layers.get_all_param_values(model)
	filename = os.path.join('./', filename)
	filename = '%s.%s' % (filename, 'npz')
	np.savez(filename, *lasagne.layers.get_all_param_values(model))

def set_param_trainability( layer, btrain=True ):
	if btrain:
		layer.params[layer.W].add('trainable')
	else:
		layer.params[layer.W].remove('trainable')

def show_params(layer):
	print(lasagne.layers.get_all_param_values(layer))

def build_mlp(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	
	l_h1 = lasagne.layers.DenseLayer( lasagne.layers.dropout(l_in, p=0.2), num_units=800, nonlinearity=lasagne.nonlinearities.rectify)
	l_h2 = lasagne.layers.DenseLayer( lasagne.layers.dropout(l_h1, p=0.5), num_units=800, nonlinearity=lasagne.nonlinearities.rectify)
	l_outclass = lasagne.layers.DenseLayer( lasagne.layers.dropout(l_h2, p=0.5), num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	return l_outclass
	

def build_cnn(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	# print(lasagne.layers.get_output_shape(l_in))
	l_c1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() )
	l_c1p = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2,2) )
	l_c2 = lasagne.layers.Conv2DLayer( l_c1p, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_le = lasagne.layers.Conv2DLayer( l_c2, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	
	# l_cclass = lasagne.layers.FlattenLayer(l_cclass, outdim=2, )
	l_outclass = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_le, p=0.5), num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	
	# print("output class:", lasagne.layers.get_output_shape(l_cclass))
	
	# print("output reconstruction:",lasagne.layers.get_output_shape(l_out))
	return l_outclass


def build_lcnn(input_var=None):
	
	print("network")
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 280, 280), input_var=input_var)
	print("\t", lasagne.layers.get_output_shape(l_in))
	
	l_c1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal(), pad="same" )
	print("\t", lasagne.layers.get_output_shape(l_c1))
	
	l_c1p = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2,2) )
	print("\t", lasagne.layers.get_output_shape(l_c1p))
	
	l_c2 = lasagne.layers.Conv2DLayer( l_c1p, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal(), pad='same' )
	print("\t", lasagne.layers.get_output_shape(l_c2))
	
	l_c2p = lasagne.layers.MaxPool2DLayer(l_c2, pool_size=(2,2) )
	print("\t", lasagne.layers.get_output_shape(l_c2p))
	
	l_c3 = lasagne.layers.Conv2DLayer( l_c2p, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal(), pad='same' )
	print("\t", lasagne.layers.get_output_shape(l_c3))

# 	l_c3p = lasagne.layers.MaxPool2DLayer(l_c3, pool_size=(2,2) )
# 	print("\t", lasagne.layers.get_output_shape(l_c3p))
	
	l_out = lasagne.layers.Conv2DLayer( l_c3, num_filters=1, filter_size=(7,7), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal(), pad='same' )
	print("\t", lasagne.layers.get_output_shape(l_out))
# 	print("\t", lasagne.layers.get_output_shape(l_out))
	
# 	l_d2 = lasagne.layers.Deconv2DLayer( l_leu, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
# 	l_d2u = lasagne.layers.Upscale2DLayer( l_d2, scale_factor=2, mode='repeat')
# 	l_d1 = lasagne.layers.Deconv2DLayer( l_d2u, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() )
# 	l_outseg = lasagne.layers.FeaturePoolLayer( l_d1, 32, pool_function=theano.tensor.max )
	
	return l_out

def build_lrae( input_var=None ):
	
	print("network")
	
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 280, 280), input_var=input_var)
	print("\t", lasagne.layers.get_output_shape(l_in))
	
	l_c1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() , pad='same')
	print("\t", lasagne.layers.get_output_shape(l_c1))
	
	l_c1p = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2,2) )
	print("\t", lasagne.layers.get_output_shape(l_c1p))
	
	l_c2 = lasagne.layers.Conv2DLayer( l_c1p, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() , pad='same')
	print("\t", lasagne.layers.get_output_shape(l_c2))
	
	l_c2p = lasagne.layers.MaxPool2DLayer( l_c2, pool_size=(2,2) )
	print("\t", lasagne.layers.get_output_shape(l_c2p))
	
	l_le = lasagne.layers.Conv2DLayer( l_c2p, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() , pad='same')
	print("\t", lasagne.layers.get_output_shape(l_le))
	
	l_leu = lasagne.layers.Deconv2DLayer( l_le, num_filters=32, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal(), crop='same')
	print("\t", lasagne.layers.get_output_shape(l_leu))
	
	l_d2u = lasagne.layers.Upscale2DLayer( l_leu, scale_factor=2, mode='repeat')
	print("\t", lasagne.layers.get_output_shape(l_d2u))
	
	l_d2 = lasagne.layers.Deconv2DLayer( l_d2u, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() , crop='same')
	print("\t", lasagne.layers.get_output_shape(l_d2))
	
	l_d1u = lasagne.layers.Upscale2DLayer( l_d2, scale_factor=2, mode='repeat')
	print("\t", lasagne.layers.get_output_shape(l_d1u))
	
	l_out = lasagne.layers.Deconv2DLayer( l_d1u, 10, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() , crop='same')
	print("\t", lasagne.layers.get_output_shape(l_out))
	
# 	l_out = lasagne.layers.FeaturePoolLayer( l_d1, 32, pool_function=theano.tensor.max )
# 	print("\t", lasagne.layers.get_output_shape(l_out))
	
	return l_out


def build_rcae(input_var=None):
	
	print("network")
	
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 280, 280), input_var=input_var)
	print("\t", lasagne.layers.get_output_shape(l_in))
	
	l_c1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() , pad='same')
	print("\t", lasagne.layers.get_output_shape(l_c1))
	
	l_c1p = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2,2) )
	print("\t", lasagne.layers.get_output_shape(l_c1p))
	
	l_c2 = lasagne.layers.Conv2DLayer( l_c1p, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() , pad='same')
	print("\t", lasagne.layers.get_output_shape(l_c2))
	
	l_c2p = lasagne.layers.MaxPool2DLayer( l_c2, pool_size=(2,2) )
	print("\t", lasagne.layers.get_output_shape(l_c2p))
	
	l_le = lasagne.layers.Conv2DLayer( l_c2p, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() , pad='same')
	print("\t", lasagne.layers.get_output_shape(l_le))
	
	l_leu = lasagne.layers.Deconv2DLayer( l_le, num_filters=32, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal(), crop='same')
	print("\t", lasagne.layers.get_output_shape(l_leu))
	
	l_d2u = lasagne.layers.Upscale2DLayer( l_leu, scale_factor=2, mode='repeat')
	print("\t", lasagne.layers.get_output_shape(l_d2u))
	
	l_d2 = lasagne.layers.Deconv2DLayer( l_d2u, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() , crop='same')
	print("\t", lasagne.layers.get_output_shape(l_d2))
	
	l_d1u = lasagne.layers.Upscale2DLayer( l_d2, scale_factor=2, mode='repeat')
	print("\t", lasagne.layers.get_output_shape(l_d1u))
	
	l_d1 = lasagne.layers.Deconv2DLayer( l_d1u, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() , crop='same')
	print("\t", lasagne.layers.get_output_shape(l_d1))
	
	l_out = lasagne.layers.FeaturePoolLayer( l_d1, 32, pool_function=theano.tensor.max )
	print("\t", lasagne.layers.get_output_shape(l_out))
	
	return l_out




def build_lae(input_var=None):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
	# print(lasagne.layers.get_output_shape(l_in))
	l_c1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() )
	l_c1p = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2,2) )
	l_c2 = lasagne.layers.Conv2DLayer( l_c1p, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_le = lasagne.layers.Conv2DLayer( l_c2, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_leu = lasagne.layers.Deconv2DLayer( l_le, num_filters=32, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_d2 = lasagne.layers.Deconv2DLayer( l_leu, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_d2u = lasagne.layers.Upscale2DLayer( l_d2, scale_factor=2, mode='repeat')
	l_d1 = lasagne.layers.Deconv2DLayer( l_d2u, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() )
	l_out = lasagne.layers.FeaturePoolLayer( l_d1, 32, pool_function=theano.tensor.max )
	l_cclass = lasagne.layers.Conv2DLayer( l_le, num_filters=1, filter_size=(8,8), nonlinearity=lasagne.nonlinearities.softmax )
	
	# l_cclass = lasagne.layers.FlattenLayer(l_cclass, outdim=2, )
	# l_outclass = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_le, p=0.5), num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	
	print("output LE:", lasagne.layers.get_output_shape(l_le))
	
	print("output reconstruction:",lasagne.layers.get_output_shape(l_out))
	return l_out, l_le

def build_mlp_output(input_var=None):
	l_le = lasagne.layers.InputLayer(shape=(None, 16, 8, 8), input_var=input_var)
	l_outclass = lasagne.layers.DenseLayer(l_le, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	print("input classifier:",lasagne.layers.get_output_shape(l_le))
	print("output classifier:",lasagne.layers.get_output_shape(l_outclass))
	return l_outclass

def build_cnn_output(input_var=None):
	l_le = lasagne.layers.InputLayer(shape=(None, 16, 8, 8), input_var=input_var)
	l_leu = lasagne.layers.Deconv2DLayer( l_le, num_filters=32, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_d2 = lasagne.layers.Deconv2DLayer( l_leu, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotNormal() )
	l_d2u = lasagne.layers.Upscale2DLayer( l_d2, scale_factor=2, mode='repeat')
	l_d1 = lasagne.layers.Deconv2DLayer( l_d2u, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotNormal() )
	l_outseg = lasagne.layers.FeaturePoolLayer( l_d1, 32, pool_function=theano.tensor.max )
	print("output segmentation:",lasagne.layers.get_output_shape(l_outseg))
	return l_outseg

"""
simple copy of the function iterate_minibatches(...) of the lasagne/examples/mnist.pyo
"""

def iterate_minibatches(inputs, segmentations, batchsize, shuffle=False):
	assert len(inputs) == len(segmentations)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], segmentations[excerpt]

def iterate_minibatches(inputs, segmentations, batchsize, shuffle=False):
	assert len(inputs) == len(segmentations)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], segmentations[excerpt]

def iterate_windows(inputs, segmentations, windowsize=[7,7]):
	assert len(inputs) == len(segmentations)
	for sx in range(0, inputs.shape[2], windowsize[0]):
		for sy in range(0, intmp.shape[3], windowsize[1]):
			i = inputs[:,:,sx:sx+windowsize[0],sy:sy+windowsize[1]]
			s = segmentations[:,:,sx:sx+windowsize[0],sy:sy+windowsize[1]]
			yield i, s


############## MAIN ################

def main():
	parser = argparse.ArgumentParser(description="experiments of supervision rate on dNN classifier")
	parser.add_argument("-o", "-output-file", 
					dest="output_file",
					default="output.npz",
					help ="filename of the output file (warning: file will be saved as a npz)")
	parser.add_argument("-e", "--number-epoch",
					dest="num_epochs",
					type=int,
					default=10,
					help="number of epoch",)
	parser.add_argument("-n", "--number-experiments",
					dest="num_exp",
					type=int,
					default=10,
					help="number of experiments",)
	parser.add_argument("-p", "--validation-proportion",
					dest="prop_valid",
					type=int,
					default=20,
					help="proportion of validation data for the split validation/train data (in %)",)
	parser.add_argument("-b", "--size-minibatch",
					dest="size_minibatch",
					type=int,
					default=10,
					help="size of minibatch",)
	parser.add_argument("-q",
					dest="set_sr", 
					help="sequence of supervisation rate (replace [-s] args if set)", 
					nargs="+",
					type=int)
	parser.add_argument('-s', '--supervisation-rate-sequence', 
					dest="seq_sr",
					default=[100 ,0 ,10],
					nargs=3,
					help='supervisation rate sequence (in%) defau ', 
					type=int)
	parser.add_argument("-u", "--bypass-autoencoded", action="store_true",dest="no_ul",
						help="bypass autoencoder (unsupervised learning)")
	parser.set_defaults(visual=False)
	

	args = parser.parse_args()

	num_epochs=args.num_epochs
	num_exp=args.num_exp
	prop_valid=args.prop_valid
	size_minibatch = args.size_minibatch 

	print("Set network")
	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')
	class_var = T.ivector('classes')
	seg_var = T.tensor4('segmentations')
	
# 	network_lcnn = build_lcnn(input_var)
	network_lcnn = build_lrae(input_var)

	# definition of what is "train" for segmentation
	out_lcnn = lasagne.layers.get_output(network_lcnn)
	out_vec = out_lcnn.reshape( ( out_lcnn.shape[0]*out_lcnn.shape[2]*out_lcnn.shape[3] , out_lcnn.shape[1]  ) )
	seg_var_vec = seg_var.reshape( ( seg_var.shape[0] * seg_var.shape[2] * seg_var.shape[3] , seg_var.shape[1]  ) )
	loss_lcnn = lasagne.objectives.categorical_crossentropy(out_lcnn, seg_var_vec)
	mloss_lcnn = lasagne.objectives.aggregate(loss_lcnn)
	params_lcnn = lasagne.layers.get_all_params(network_lcnn, trainable=True)
	updates_lcnn = lasagne.updates.nesterov_momentum(mloss_lcnn, params_lcnn, learning_rate=0.1, momentum=0.9)
	train_fn_lcnn = theano.function([input_var, seg_var], mloss_lcnn, updates=updates_lcnn, allow_input_downcast=True)
	
	
# 	# definition of Evaluation for segmentation
	segmentation = lasagne.layers.get_output(network_lcnn, deterministic=True)
	segmentation_vec = segmentation.reshape( ( segmentation.shape[0]*segmentation.shape[2]*segmentation.shape[3] , segmentation.shape[1]  ) )
	segmentation_loss = lasagne.objectives.categorical_crossentropy(segmentation_vec, seg_var_vec)
	segmentation_mloss = lasagne.objectives.aggregate(segmentation_loss)
	eval_lcnn = theano.function([input_var, seg_var], [segmentation_mloss, segmentation] , allow_input_downcast=True)
# 	
# 	overall_time = time.time()
# 	
	# mnist dataset
	print("Loading mnist data...")
	X_train, y_train, X_test, y_test = load_dataset_mnist()
	
# 	print( X_train.shape )
	X_trainup = np.append( X_train, np.zeros( ( 2*len(X_train), 1, 28,28 ) ).astype(int), axis=0 )
	X_trainup = np.tile(X_trainup, (1,1,1,1) )
# 	Y_trainup = np.append( np.ones(y_train.shape), np.zeros( ( 2*len(y_train) ) ).astype(int), axis=0 )
	Y_trainup = np.append( 1+y_train, np.zeros( ( 2*len(y_train) ) ).astype(int), axis=0 )
	Y_trainup = np.tile(Y_trainup, (1))
	X_testup = np.append( X_test, np.zeros( ( 2*len(X_test), 1, 28,28 ) ).astype(int), axis=0 )
	X_testup = np.tile(X_testup, (1,1,1,1))
# 	Y_testup = np.append( np.ones(y_test.shape), np.zeros( ( 2*len(y_test) ) ).astype(int), axis=0 )
	Y_testup = np.append( 1+y_test, np.zeros( ( 2*len(y_test) ) ).astype(int), axis=0 )
	Y_testup = np.tile(Y_testup, (1))

	print( X_trainup.shape )
	print( Y_trainup.shape )
	print( X_testup.shape )
	print( Y_testup.shape )
	
	
# 	print(z.shape)
	indices = np.arange( len(X_trainup) )
	
	np.random.shuffle(indices)
	
	n = 28 
	m = 10
	
	X_trainup = X_trainup[ indices ]
	X_trainup = X_trainup.reshape( -1 , 1, 28*m, 28 )
	X_trainup = np.swapaxes(X_trainup, 2,3)
	X_trainup = X_trainup.reshape( -1 , 1, 28*m, 28*m )
	X_trainup = np.swapaxes(X_trainup, 2,3)
	
	Y_trainup = Y_trainup
	Y_trainup = Y_trainup[indices]
	
	Y_trainup = Y_trainup.reshape( -1 , 1, m, m )
	Y_trainup = np.repeat( Y_trainup, n, axis=2 )
# 	Y_trainup = Y_trainup.reshape( -1 , 1, 10, 10 )
	Y_trainup = np.repeat( Y_trainup, n, axis=3 )
# 	Y_trainup = Y_trainup.reshape( -1 , 1, 70, 70 )
# 	Y_trainup = Y_trainup.reshape( -1 , 1, 280, 280 )
	Y_trainup = np.swapaxes(Y_trainup, 2,3)
# 	Y_trainup = Y_trainup.T
# 	Y_trainup = Y_trainup.reshape( -1 , 1, 280, 28 )
# 	Y_trainup = np.swapaxes(Y_trainup, 2,3)
# 	Y_trainup = Y_trainup.reshape( -1 , 1, 280, 280 )
# 	Y_trainup = np.swapaxes(Y_trainup, 2,3)
	
	indicesTest = np.arange( len(X_testup) )
	
	np.random.shuffle(indicesTest)

	X_testup = X_testup[indicesTest]
	X_testup = X_testup.reshape( -1 , 1, 28*m, 28 )
	X_testup = np.swapaxes(X_testup, 2,3)
	X_testup = X_testup.reshape( -1 , 1, 28*m, 28*m )
	X_testup = np.swapaxes(X_testup, 2,3)
	
<<<<<<< HEAD
	Y_testup = Y_testup/10
	Y_testup = Y_testup[indicesTest]
	Y_testup = Y_testup.reshape( -1 , 1, m, m )
	Y_testup = np.repeat( Y_testup, n, axis=2 )
	Y_testup = np.repeat( Y_testup, n, axis=3 )
# 	Y_testup = Y_testup.reshape( -1 , 1, 280, 280 )
	Y_testup = np.swapaxes(Y_testup, 2,3)
# 	Y_testup = np.repeat( Y_testup, 10*10 )
	
=======
	y_trainup = y_trainup[indices]
	y_trainup = np.repeat( y_trainup, 10*10 )
	y_trainup = y_trainup.reshape( -1 , 1, 1, 1 )
	y_trainup = y_trainup.reshape( -1 , 1, 28, 28 )
# 	y_trainup = np.swapaxes(y_trainup, 2,3)
# 	y_trainup = y_trainup.reshape( -1 , 1, 280, 280 )
# 	y_trainup = np.swapaxes(y_trainup, 2,3)
	
# 	y_testup = np.repeat( y_testup, 10*10 )
	
	print( y_trainup.shape )
>>>>>>> 5e8c67d1602e8c7e77086650234d85a0694959d6
	print( X_trainup.shape )
	print( Y_trainup.shape )
	print( X_testup.shape )
	print( Y_testup.shape )
# 	i = np.random.randint(0, len(X_trainup))
# 	print( ( stats.threshold(X_trainup[ i ,0], threshmax=0.1, newval=1 ) ).reshape((28,28)).astype(int) )
# 	print(Y_trainup[ 3 ,0])
# 	plt.imshow(X_trainup[ 3 ,0])
# 	plt.savefig('temp.png')
# 	print( args.seq_sr )
	
	# argument recuperation
	
	X_train, Y_train, X_valid, Y_valid = split_data(X_trainup, Y_trainup, 20)
	X_test = X_testup.astype(float)
	Y_test = Y_testup.astype(float)
	
	overall_time = time.time()
	for e in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, Y_train, size_minibatch, shuffle=True):
			inputs, segmentations = batch
			train_err += train_fn_lcnn(inputs, segmentations)
			train_batches += 1
		
		AceTrain = (train_err / train_batches)

		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_valid, Y_valid, size_minibatch, shuffle=False):
			inputs, segmentations = batch
			err, seg = eval_lcnn(inputs, segmentations)
			val_err += err
			val_batches += 1
		
		AceValid = (val_err / val_batches)
		t = time.time() - overall_time
		hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
		
		print("-----Training-----")
		print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
		print("")	
		print("Epoch: ", e + 1, "/", num_epochs, "\tt: {:.3f}s".format( time.time() - start_time))
		print("\t Training Loss:\t{:.6f} ".format( AceTrain ) )
		print("\t Validation Loss:\t{:.6f} ".format( AceValid ) )
# 		print("\t validation recons MSE:\t{:.6f}".format( MseReconVal ) )
		print("")

	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, Y_test, size_minibatch, shuffle=False):
		inputs, segmentations = batch
		err, seg = eval_lcnn(inputs, segmentations)
		test_err += err
		# test_acc += acc
		
		test_batches += 1
		
	
	AceTest = (test_err / test_batches)
	t = time.time() - overall_time
	hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
	
	print("-----Test-----")
	print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
	print("")	
	print("\t Test Loss:\t{:.6f} ".format( AceTest ) )
# 		print("\t validation recons MSE:\t{:.6f}".format( MseReconVal ) )
	print("-----------")
	i = np.random.randint(0, len(seg))
	plt.imshow(seg[ i ,0], cmap = plt.get_cmap('gray'))
	plt.savefig('temp.png')
	plt.imshow(segmentations[ i ,0], cmap = plt.get_cmap('gray'))
	plt.savefig('temp2.png')

if __name__ == "__main__":
	print("guided fully-connected convolutional autoencoder")
	main()
	

