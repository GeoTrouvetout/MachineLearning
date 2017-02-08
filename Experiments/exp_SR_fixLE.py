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

def split_data(X, y, Y, proportion=20):
	T_ind = np.arange(len(y))
	np.random.shuffle(T_ind)
	X = X[T_ind]
	y = y[T_ind]
	Y = Y[T_ind]
	
	nb_p = np.floor( ( proportion/100 ) * len(X) ).astype(int)  # part used for the supervised learning
	nb_ip = np.floor( (1-(proportion/100)) * len(X) ).astype(int)  # part used for the autoencoder (the rest for training the classifier)
	if nb_p !=0:
		# supervised / non-supervised split
		X_ip, X_p = X[:-nb_p], X[-nb_p:]
		y_ip, y_p = y[:-nb_p], y[-nb_p:]
		Y_ip, Y_p = Y[:-nb_p], Y[-nb_p:]
	elif nb_p == 0: # if supervision Rate 100% -> p = 0 -> the selection of indices [-0:] and [:-0] are permuted
		X_p, X_ip = X[:-nb_p], X[-nb_p:]
		y_p, y_ip = y[:-nb_p], y[-nb_p:]
		Y_p, Y_ip = Y[:-nb_p], Y[-nb_p:]

	return X_p, y_p, Y_p, X_ip, y_ip, Y_ip



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
# 	l_cclass = lasagne.layers.Conv2DLayer( l_le, num_filters=10, filter_size=(8,8), nonlinearity=lasagne.nonlinearities.softmax )

	# l_cclass = lasagne.layers.FlattenLayer(l_cclass, outdim=2, )
	# l_outclass = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_le, p=0.5), num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	
	print("output LE:", lasagne.layers.get_output_shape(l_le))
	
	print("output reconstruction:",lasagne.layers.get_output_shape(l_out))
	return l_out, l_le

def build_mlp_output(input_var=None):
	l_le = lasagne.layers.InputLayer(shape=(None, 16, 8, 8), input_var=input_var)
# 	l_h = lasagne.layers.DenseLayer(l_le, num_units=100, nonlinearity=lasagne.nonlinearities.sigmoid)
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

def iterate_minibatches(inputs, targets, classes, segmentations, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt], classes[excerpt], segmentations[excerpt]


############## MAIN ################

def main():
	parser = argparse.ArgumentParser(description="experiments of supervision rate on dNN classifier")
	parser.add_argument("outnpz", help ="filename of the output file (warning: file will be saved as a npz)")
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
					default=1000,
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
	
	network_mlp = build_mlp(input_var)
	network_cnn = build_cnn(input_var)
	network_recon, network_le = build_lae(input_var)
	out_le = lasagne.layers.get_output( network_le, deterministic=True )
	network_class = build_mlp_output(out_le)
	network_seg = build_cnn_output(out_le)


	params_init_network_recon = lasagne.layers.get_all_param_values(network_recon)
	params_init_network_class = lasagne.layers.get_all_param_values(network_class)
	params_init_network_le = lasagne.layers.get_all_param_values(network_le)
	params_init_network_seg = lasagne.layers.get_all_param_values(network_seg)

	print("number of params for recon",lasagne.layers.count_params(network_recon))
	print("number params for class ",lasagne.layers.count_params(network_class))
	print("number of params for seg",lasagne.layers.count_params(network_seg))
	print("number params for LE ",lasagne.layers.count_params(network_le))
	
	
	
	# definition of what is "train" for classicals MLP 
	out_mlp = lasagne.layers.get_output(network_mlp)
	ce_mlp = lasagne.objectives.categorical_crossentropy(out_mlp, class_var)
	ace_mlp = lasagne.objectives.aggregate(ce_mlp)
	params_mlp = lasagne.layers.get_all_params(network_mlp, trainable=True)
	updates_mlp = lasagne.updates.nesterov_momentum(ace_mlp, params_mlp, learning_rate=0.1, momentum=0.9)
	train_fn_mlp = theano.function([input_var, class_var], ace_mlp, updates=updates_mlp)

	# definition of what is "train" for classicals MLP and CNN
	out_cnn = lasagne.layers.get_output(network_cnn)
	ce_cnn = lasagne.objectives.categorical_crossentropy(out_cnn, class_var)
	ace_cnn = lasagne.objectives.aggregate(ce_cnn)
	params_cnn = lasagne.layers.get_all_params(network_cnn, trainable=True)
	updates_cnn = lasagne.updates.nesterov_momentum(ace_cnn, params_cnn, learning_rate=0.1, momentum=0.9)
	train_fn_cnn = theano.function([input_var, class_var], ace_cnn, updates=updates_cnn)

	# definition of what is "train" for reconstruction
	out_recon = lasagne.layers.get_output(network_recon)
	se_recon = lasagne.objectives.squared_error(out_recon, target_var)
	mse_recon = lasagne.objectives.aggregate(se_recon)
	params_recon = lasagne.layers.get_all_params(network_recon, trainable=True)
	updates_recon = lasagne.updates.nesterov_momentum(mse_recon, params_recon, learning_rate=0.1, momentum=0.9)
	train_fn_recon = theano.function([input_var, target_var], mse_recon, updates=updates_recon)
	
	# definition of what is "train" for classifier
	out_class = lasagne.layers.get_output(network_class)
	ce_class = lasagne.objectives.categorical_crossentropy(out_class, class_var)
	ace_class = lasagne.objectives.aggregate(ce_class)
	params_class = lasagne.layers.get_all_params(network_class, trainable=True)
	updates_class = lasagne.updates.nesterov_momentum(ace_class, params_class, learning_rate=0.1, momentum=0.9)
	train_fn_class = theano.function([out_le, class_var], ace_class, updates=updates_class)
	

	# definition of what is "train" for segmentation
	out_seg = lasagne.layers.get_output(network_seg)
	se_seg = lasagne.objectives.squared_error(out_seg, seg_var)
	mse_seg= lasagne.objectives.aggregate(se_seg)
	params_seg = lasagne.layers.get_all_params(network_seg, trainable=True)
	updates_seg = lasagne.updates.nesterov_momentum(mse_seg, params_seg, learning_rate=0.1, momentum=0.9)
	train_fn_seg = theano.function([out_le, seg_var], mse_seg, updates=updates_seg)

	# definition of Evaluation for mlp
	mlp = lasagne.layers.get_output(network_mlp, deterministic=True)
	mlp_ce = lasagne.objectives.categorical_crossentropy(mlp, class_var)
	mlp_ace = lasagne.objectives.aggregate(mlp_ce)
	mlp_acc = T.mean( T.eq( T.argmax( mlp, axis=1 ), class_var ), dtype=theano.config.floatX )
	eval_mlp = theano.function([input_var, class_var], [mlp_ace, mlp_acc] )

	# definition of Evaluation for cnn
	cnn = lasagne.layers.get_output(network_cnn, deterministic=True)
	cnn_ce = lasagne.objectives.categorical_crossentropy(cnn, class_var)
	cnn_ace = lasagne.objectives.aggregate(cnn_ce)
	cnn_acc = T.mean( T.eq( T.argmax( cnn, axis=1 ), class_var ), dtype=theano.config.floatX )
	eval_cnn = theano.function([input_var, class_var], [cnn_ace, cnn_acc] )
	
	# definition of Evaluation for classification
	classification = lasagne.layers.get_output(network_class, deterministic=True)
	classification_ce = lasagne.objectives.categorical_crossentropy(classification, class_var)
	classification_ace = lasagne.objectives.aggregate(classification_ce)
	classification_acc = T.mean( T.eq( T.argmax( classification, axis=1 ), class_var ), dtype=theano.config.floatX )
	eval_class = theano.function([out_le, class_var], [classification_ace, classification_acc] )

	# definition of Evaluation for reconstruction
	reconstruction = lasagne.layers.get_output(network_recon, deterministic=True)
	reconstruction_se = lasagne.objectives.squared_error(reconstruction, target_var)
	reconstruction_mse = lasagne.objectives.aggregate(reconstruction_se)
	eval_recon = theano.function([input_var, target_var], reconstruction_mse)
	
	# definition of Evaluation for segmentation
	segmentation = lasagne.layers.get_output(network_seg, deterministic=True)
	segmentation_se = lasagne.objectives.squared_error(segmentation, seg_var)
	segmentation_mse = lasagne.objectives.aggregate(segmentation_se)
	eval_seg = theano.function([out_le, seg_var], segmentation_mse )
	
	# get Local Encoder (LE) features
	eval_le = theano.function([input_var], out_le )
	
	overall_time = time.time()
	# mnist dataset
	print("Loading mnist data...")
	X_train, y_train, X_test, y_test = load_dataset_mnist()
	
	Y_train = np.repeat(y_train,28*28).reshape((len(y_train), 1, 28, 28))
	Y_test = np.repeat(y_test,28*28).reshape((len(y_test), 1, 28, 28))

	# displayImage(Y_test[np.random.randint(0, len(Y_test)),0,:,:]) # display random image
	
	print( Y_train.shape )
	print( X_train.shape )
	print( Y_test.shape )
	print( X_test.shape )

# 	print( args.seq_sr )

	# argument recuperation

	if not args.set_sr:
		srmax = args.seq_sr[0] 
		srstep = args.seq_sr[2] 
		srmin = args.seq_sr[1] - srstep 
		seqm = np.arange(srmax, srmin, -srstep) 
	else:
		seqm = args.set_sr
	print("################")
	print(seqm)
	print("################")
	seqn = np.arange(num_exp) 

	### results for Reconstruction ###
	OptNbSample_recon = np.zeros ( [len(seqm), len(seqn)] )
	OptNbEpoch_recon = np.zeros ( [len(seqm), len(seqn)] )

	OptMseTrain_recon = np.zeros ( [len(seqm), len(seqn)] )
	OptMseValid_recon = np.zeros ( [len(seqm), len(seqn)] )
	TensorMseTrain_recon = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorMseValid_recon = np.zeros( [len(seqm), len(seqn), num_epochs] )

	### results for Classification ###
	OptNbSample_mlp = np.zeros ( [len(seqm), len(seqn)] )
	OptNbEpoch_mlp = np.zeros ( [len(seqm), len(seqn)] )
	OptNbSample_cnn = np.zeros ( [len(seqm), len(seqn)] )
	OptNbEpoch_cnn = np.zeros ( [len(seqm), len(seqn)] )
	OptNbSample_class = np.zeros ( [len(seqm), len(seqn)] )
	OptNbEpoch_class = np.zeros ( [len(seqm), len(seqn)] )

	OptAceTrain_cnn = np.zeros( [len(seqm), len(seqn)] ) 
	OptAccValid_cnn = np.zeros( [len(seqm), len(seqn)] ) 
	OptAceValid_cnn = np.zeros( [len(seqm), len(seqn)] ) 
	TensorAceTrain_cnn = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAceValid_cnn = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAccValid_cnn = np.zeros( [len(seqm), len(seqn), num_epochs] )

	OptAceTrain_mlp = np.zeros( [len(seqm), len(seqn)] ) 
	OptAccValid_mlp = np.zeros( [len(seqm), len(seqn)] ) 
	OptAceValid_mlp = np.zeros( [len(seqm), len(seqn)] ) 
	TensorAceTrain_mlp = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAceValid_mlp = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAccValid_mlp = np.zeros( [len(seqm), len(seqn), num_epochs] )

	OptAceTrain_class = np.zeros( [len(seqm), len(seqn)] ) 
	OptAccValid_class = np.zeros( [len(seqm), len(seqn)] ) 
	OptAceValid_class = np.zeros( [len(seqm), len(seqn)] ) 
	TensorAceTrain_class = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAceValid_class = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAccValid_class = np.zeros( [len(seqm), len(seqn), num_epochs] )

	### results TEST Classification/Reconstruction ###
	ArrayAccTest_mlp = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayAceTest_mlp = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayAccTest_cnn = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayAceTest_cnn = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayAccTest_class = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayAceTest_class = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayMseTest_recon = np.zeros ( [len(seqm), len(seqn)] )
	
	for m in np.arange(len(seqm)) :
		
		prop_train_s = seqm[m] 
		print("learning supervision rate", prop_train_s,"%" )
		for n in seqn:
			print("re-initialize network parameters ... ")
# 			lasagne.layers.set_all_param_values( network_enc, params_init_network_enc )
# 			lasagne.layers.set_all_param_values( network_class, params_init_network_class )
# 			print("#######")
# # 			show_params(network_le)
# # 			show_params(network_class)
# 			print("#######")
				
			print("experiment:", n+1, "/", len(seqn))
			T_ind = np.arange(len(y_train))
			np.random.shuffle(T_ind)
			X = X_train[T_ind]
			y = y_train[T_ind]
			Y = Y_train[T_ind]
			
			if prop_valid <= 0 or prop_valid>=100 :
				print("warning : validation/Training proportion cannot be 0% or 100%")
				prop_valid=20
			
			nb_train_s = np.floor( ( prop_train_s/100 ) * len(X_train) ).astype(int)  # part used for the supervised learning
			nb_train_ns = np.floor( (1-(prop_train_s/100)) * len(X_train) ).astype(int)  # part used for the autoencoder (the rest for training the classifier)

			X_s, y_s, Y_s, X_ns, y_ns, Y_ns = split_data( X, y, Y, prop_train_s ) 

			# print("number images for supervised learning (train/val):", nb_train_s , "(", len(X_train_s), "/", len(X_valid_s),")")

			# print("number images for non-supervised learning (train/val):", nb_train_ns , "(", len(X_train_ns), "/", len(X_valid_ns),")")
# 

			nb_valid_s = np.floor( (prop_valid/100)* len(X_s) ).astype(int)
			nb_valid_ns = np.floor( (prop_valid/100)* len(X_ns) ).astype(int)

			X_valid_s, y_valid_s, Y_valid_s, X_train_s, y_train_s, Y_train_s = split_data(X_s, y_s, Y_s, prop_valid )

			X_valid_ns, y_valid_ns, Y_valid_ns, X_train_ns, y_train_ns, Y_train_ns = split_data(X_ns, y_ns, Y_ns, prop_valid )

			print("number images for supervised learning (train/val):", nb_train_s , "(", len(X_train_s), "/", len(X_valid_s),")")

			print("number images for non-supervised learning (train/val):", nb_train_ns , "(", len(X_train_ns), "/", len(X_valid_ns),")")

			
# # 			print(len(X_train_class), len(X_train_enc))
			
			###############
			# iteration over epochs:
			training_time = time.time()
			MseReconTrain_lowest = sys.float_info.max
			MseReconVal_lowest = sys.float_info.max
			best_nbsample = 0
			params_recon_best = lasagne.layers.get_all_param_values(network_recon)
# 			if not args.no_ul:
			if not len(y_train_ns) ==0:
				for e_ns in range(num_epochs):
						
					train_recon_mse = 0
					train_batches = 0
					val_recon_mse = 0
					val_batches = 0
					start_time = time.time()
						
					### shuffle indices of train/valid data
						
					ind_train_ns = np.arange( len(y_train_ns) )
					np.random.shuffle( ind_train_ns )
					X_train_ns = X_train_ns[ ind_train_ns ]
					y_train_ns = y_train_ns[ ind_train_ns ]
					Y_train_ns = Y_train_ns[ ind_train_ns ]
					
					#### batch TRAIN ENCODER ####
					for batch in iterate_minibatches(X_train_ns, X_train_ns, y_train_ns, Y_train_ns, size_minibatch, shuffle=True):
						inputs, targets, classes, segmentations = batch
						train_recon_mse += train_fn_recon( inputs, targets )
						train_batches += 1
					
					MseReconTrain = 0
					if train_batches != 0: 
						MseReconTrain = (train_recon_mse / train_batches)

	# 					#### batch VALID ENCODER ####
					for batch in iterate_minibatches(X_valid_ns, X_valid_ns, y_valid_ns, Y_valid_ns, size_minibatch, shuffle=True):
						inputs, targets, classes, segmentations = batch
						val_recon_mse += eval_recon(inputs, targets)
						val_batches += 1
					
					MseReconVal = 0
					if val_batches != 0: 
						MseReconVal = (val_recon_mse / val_batches)
					t = time.time() - overall_time
					hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
					print("-----UnSupervised-----")
					print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
					print("")	
					print("Epoch: ", e_ns + 1, "/", num_epochs, "\tn: %d/%d" % (n+1,len(seqn)), "\tt: {:.3f}s".format( time.time() - start_time), "\ts: %d" %(prop_train_s), "%")
					print("\t training recons MSE:\t{:.6f} ".format( MseReconTrain ) )
					print("\t validation recons MSE:\t{:.6f}".format( MseReconVal ) )
					print("")
					
					TensorMseTrain_recon[m][n][e_ns] = MseReconTrain
					TensorMseValid_recon[m][n][e_ns] = MseReconVal
						
					if MseReconVal < MseReconVal_lowest:
						MseReconVal_lowest = MseReconVal
						OptMseTrain_recon[m][n] = MseReconTrain
						OptMseValid_recon[m][n] = MseReconVal
						OptNbSample_recon[m][n] = e_ns * len(X_train_ns)
						OptNbEpoch_recon[m][n] = e_ns
						params_recon_best = lasagne.layers.get_all_param_values(network_recon)
						# params_nn_s_best = lasagne.layers.get_all_param_values(network_class)
						
					# lasagne.layers.set_all_param_values( network_enc, params_init_network_enc )
	# 				# lasagne.layers.set_all_param_values( network_class, params_init_network_class )
	# # 				lasagne.layers.set_all_param_values( network_enc, params_nn_ns_best )
				lasagne.layers.set_all_param_values( network_recon, params_recon_best )

			AccCNNTrain_highest = sys.float_info.min
			AccCNNVal_highest = sys.float_info.min
			AccMLPTrain_highest = sys.float_info.min
			AccMLPVal_highest = sys.float_info.min
			AccClassTrain_highest = sys.float_info.min
			AccClassVal_highest = sys.float_info.min
			best_nbsample = 0
			
			
			
# 			# lasagne.layers.set_all_param_values( network_class, params_init_network_class )
			params_cnn_best = lasagne.layers.get_all_param_values(network_class)
			params_mlp_best = lasagne.layers.get_all_param_values(network_class)
			params_class_best = lasagne.layers.get_all_param_values(network_class)
			
# # 			set_param_trainability(network_le, False)
			
			for e_s in range(num_epochs):
				
				train_mlp_ace= 0
				train_cnn_ace= 0
				train_class_ace= 0
				val_mlp_ace = 0
				val_mlp_acc = 0
				val_cnn_ace = 0
				val_cnn_acc = 0
				val_class_ace = 0
				val_class_acc = 0
				# val_mse = 0
				train_batches = 0
				val_batches = 0
				start_time = time.time()
				
				### shuffle indices of train/valid data
					
				ind_train_s = np.arange( len(y_train_s) )
				np.random.shuffle( ind_train_s )
				X_train_s = X_train_s[ ind_train_s ]
				y_train_s = y_train_s[ ind_train_s ]
				Y_train_s = Y_train_s[ ind_train_s ]
				
				#### batch TRAIN CLASSIFIER ####
				for batch in iterate_minibatches(X_train_s, X_train_s, y_train_s, Y_train_s, size_minibatch, shuffle=True):
					inputs, targets, classes, segmentations = batch
					le = eval_le( inputs )
					train_class_ace += train_fn_class( le, classes )
# 					paramsle = lasagne.layers.get_all_param_values(network_le)
# 					paramsclass = lasagne.layers.get_all_param_values(network_class)
# 					print("#########", np.sum(paramsle[0].reshape([1,-1])), np.sum(paramsclass[0].reshape([1,-1])) )
					train_mlp_ace += train_fn_mlp( inputs, classes )
					train_cnn_ace += train_fn_cnn( inputs, classes )
					train_batches += 1
				
				AceClassTrain = 0
				AceMLPTrain = 0
				AceCNNTrain = 0
				if train_batches != 0: 
					AceClassTrain = (train_class_ace / train_batches)
					AceMLPTrain = (train_mlp_ace / train_batches)
					AceCNNTrain = (train_cnn_ace / train_batches)

				#### batch VALID CLASSIFIER ####
				for batch in iterate_minibatches(X_valid_s, X_valid_s, y_valid_s, Y_valid_s, size_minibatch, shuffle=True):
					inputs, targets, classes, segmentations = batch
					le = eval_le( inputs )
					ace, acc = eval_cnn( inputs, classes )
					val_cnn_ace += ace
					val_cnn_acc += acc
					ace, acc = eval_mlp( inputs, classes )
					val_mlp_ace += ace
					val_mlp_acc += acc
					ace, acc = eval_class( le, classes )
					val_class_ace += ace
					val_class_acc += acc
					val_batches += 1
				
				AceMLPVal = 0
				AccMLPVal = 0
				AceCNNVal = 0
				AccCNNVal = 0
				AceClassVal = 0
				AccClassVal = 0
				if val_batches != 0: 
					AceMLPVal = (val_mlp_ace / val_batches)
					AccMLPVal = (val_mlp_acc/ val_batches)
					AceCNNVal = (val_cnn_ace / val_batches)
					AccCNNVal = (val_cnn_acc/ val_batches)
					AceClassVal = (val_class_ace / val_batches)
					AccClassVal = (val_class_acc/ val_batches)
				
				t = time.time() - overall_time
				
				hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
				print("-----Supervised-----")
				print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
				print("")	
				print("Epoch: ", e_s + 1, "/", num_epochs, "\tn: %d/%d" % (n+1,len(seqn)), "\tt: {:.3f}s".format( time.time() - start_time), "\ts: %d" %(prop_train_s), "%")
				print("\t training class ACE:\t\t{:.6f} ".format( AceClassTrain ) )
				print("\t validation class ACE:\t\t{:.6f} ".format( AceClassVal ) )
# 				print("\t validation class ACC:\t{:.6f}".format( AccClassVal ) )
				print("\t validation class ACC:\t\t{:.2f}%".format( 100*(AccClassVal) )  )
				print("\t validation MLP ACC:\t\t{:.2f} %".format( 100*(AccMLPVal) ) )
				print("\t validation CNN ACC:\t\t{:.2f} %".format( 100*(AccCNNVal) ) )
				print("")	
				
				TensorAceValid_mlp[m][n][e_s] = AceMLPVal
				TensorAccValid_mlp[m][n][e_s] = AccMLPVal
				TensorAceValid_cnn[m][n][e_s] = AceCNNVal
				TensorAccValid_cnn[m][n][e_s] = AccCNNVal
				TensorAceValid_class[m][n][e_s] = AceClassVal
				TensorAccValid_class[m][n][e_s] = AccClassVal
				
				if AccCNNVal > AccCNNVal_highest:
					AccCNNVal_highest = AccCNNVal
					OptAceTrain_cnn[m][n] = AceCNNTrain
					OptAceValid_cnn[m][n] = AceCNNVal
					OptAccValid_cnn[m][n] = AccCNNVal
					OptNbSample_cnn[m][n] = e_s * len(X_train_ns)
					OptNbEpoch_cnn[m][n] = e_s
					params_cnn_best = lasagne.layers.get_all_param_values(network_cnn)
				
				if AccMLPVal > AccMLPVal_highest:
					AccMLPVal_highest = AccMLPVal
					OptAceTrain_mlp[m][n] = AceMLPTrain
					OptAceValid_mlp[m][n] = AceMLPVal
					OptAccValid_mlp[m][n] = AccMLPVal
					OptNbSample_mlp[m][n] = e_s * len(X_train_ns)
					OptNbEpoch_mlp[m][n] = e_s
					params_mlp_best = lasagne.layers.get_all_param_values(network_mlp)
				
				if AccClassVal > AccClassVal_highest:
					AccClassVal_highest = AccClassVal
					OptAceTrain_class[m][n] = AceClassTrain
					OptAceValid_class[m][n] = AceClassVal
					OptAccValid_class[m][n] = AccClassVal
					OptNbSample_class[m][n] = e_s * len(X_train_ns)
					OptNbEpoch_class[m][n] = e_s
					params_class_best = lasagne.layers.get_all_param_values(network_class)
					# params_nn_s_best = lasagne.layers.get_all_param_values(network_class)
			
				
			###### TEST #####
			lasagne.layers.set_all_param_values( network_recon, params_recon_best )
			lasagne.layers.set_all_param_values( network_class, params_class_best )
			lasagne.layers.set_all_param_values( network_mlp, params_mlp_best )
			lasagne.layers.set_all_param_values( network_cnn, params_cnn_best )
			
			test_recon_mse = 0
			test_mlp_ace = 0
			test_mlp_acc = 0
			test_cnn_ace = 0
			test_cnn_acc = 0
			test_class_ace = 0
			test_class_acc = 0
			test_batches = 0
			
			start_time = time.time()
			
			MseReconTest = 0
			AccClassTest = 0
			AccMLPTest = 0
			AccCNNTest = 0
			AceClassTest = 0
			AceMLPTest = 0
			AceCNNTest = 0
			
			for batch in iterate_minibatches(X_test, X_test, y_test, Y_test, size_minibatch, shuffle=False):
				inputs, targets, classes, segmentations = batch
				test_recon_mse += eval_recon( inputs, targets )
				le = eval_le( inputs )
				ace, acc = eval_cnn( inputs, classes)
				test_cnn_ace += ace
				test_cnn_acc += acc
				ace, acc = eval_mlp( inputs, classes)
				test_mlp_ace += ace
				test_mlp_acc += acc
				ace, acc = eval_class( le, classes)
				test_class_ace += ace
				test_class_acc += acc
				test_batches += 1

			MseReconTest = test_recon_mse / test_batches
			AccClassTest = test_class_acc / test_batches 
			AccMLPTest = test_mlp_acc / test_batches
			AccCNNTest = test_cnn_acc / test_batches
			AceClassTest = test_class_ace / test_batches
			AceMLPTest = test_mlp_ace / test_batches
			AceCNNTest = test_cnn_ace / test_batches
			
			ArrayAccTest_mlp[m][n]= AccMLPTest 
			ArrayAceTest_mlp[m][n]= AceMLPTest
			ArrayAccTest_cnn[m][n]= AccCNNTest
			ArrayAceTest_cnn[m][n]= AceCNNTest
			ArrayAccTest_class[m][n]= AccClassTest
			ArrayAceTest_class[m][n]= AceClassTest
			ArrayMseTest_recon[m][n]= MseReconTest
			
			
	                           
# 				inputs, targets, classes = batch
# 				ace, acc = val_fn_class( inputs, classes )
# 				test_ace += ace
# 				test_acc += acc
# 				test_mse += val_fn_enc(inputs, targets)
# 				test_batches += 1
			
# 			MseTest = (test_mse / test_batches)
# 			AceTest = (test_ace / test_batches)
# 			AccTest = (test_acc / test_batches)
# 			ArrayAccTest[m][n] = AccTest
# 			ArrayAceTest[m][n] = AceTest
# 			ArrayMseTest[m][n] = MseTest
			print("################ TEST RESULTS ##################")
			print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
			print("")	
			print("Test Results |", "\tn: %d/%d" % (n+1,len(seqn)), "\tt: {:.3f}s".format( time.time() - training_time), "\ts: %d" %(prop_train_s), "%")
# 			print("Test Results | supervision rate ", prop_train_s, "% | experiment ", n, "/", len(seqn) )
			print("\t test recons MSE:\t\t{:.6f}".format( MseReconTest) )
			print("\t test class ACE:\t\t{:.6f}".format( AceClassTest) )
			print("\t test class ACC:\t\t{:.2f} %".format( 100*(AccClassTest) ) )
			print("\t test MLP ACC:\t\t\t{:.2f} %".format( 100*(AccMLPTest) ) )
			print("\t test CNN ACC:\t\t\t{:.2f} %".format( 100*(AccCNNTest) ) )
			print("################################################")
			print("")	

			np.savez(args.outnpz, 
					OptNbSample_recon=OptNbSample_recon, 
					OptNbEpoch_recon=OptNbEpoch_recon, 
					OptMseTrain_recon=OptMseTrain_recon, 
					OptMseValid_recon=OptMseValid_recon, 
					TensorMseTrain_recon=TensorMseTrain_recon, 
					TensorMseValid_recon=TensorMseValid_recon, 
					OptNbSample_mlp=OptNbSample_mlp, 
					OptNbEpoch_mlp=OptNbEpoch_mlp, 
					OptNbSample_cnn=OptNbSample_cnn, 
					OptNbEpoch_cnn=OptNbEpoch_cnn, 
					OptNbSample_class=OptNbSample_class, 
					OptNbEpoch_class=OptNbEpoch_class, 
					OptAceTrain_cnn =OptAceTrain_cnn, 
					OptAccValid_cnn =OptAccValid_cnn, 
					OptAceValid_cnn =OptAceValid_cnn, 
					TensorAceTrain_cnn=TensorAceTrain_cnn, 
					TensorAceValid_cnn=TensorAceValid_cnn, 
					TensorAccValid_cnn=TensorAccValid_cnn, 
					OptAceTrain_mlp =OptAceTrain_mlp, 
					OptAccValid_mlp =OptAccValid_mlp, 
					OptAceValid_mlp =OptAceValid_mlp, 
					TensorAceTrain_mlp=TensorAceTrain_mlp, 
					TensorAceValid_mlp=TensorAceValid_mlp, 
					TensorAccValid_mlp=TensorAccValid_mlp, 
					OptAceTrain_class =OptAceTrain_class, 
					OptAccValid_class =OptAccValid_class, 
					OptAceValid_class =OptAceValid_class, 
					TensorAceTrain_class=TensorAceTrain_class, 
					TensorAceValid_class=TensorAceValid_class, 
					TensorAccValid_class=TensorAccValid_class, 
					ArrayAccTest_mlp =ArrayAccTest_mlp, 
					ArrayAceTest_mlp =ArrayAceTest_mlp, 
					ArrayAccTest_cnn =ArrayAccTest_cnn, 
					ArrayAceTest_cnn =ArrayAceTest_cnn, 
					ArrayAccTest_class =ArrayAccTest_class, 
					ArrayAceTest_class =ArrayAceTest_class, 
					ArrayMseTest_recon=ArrayMseTest_recon )

			
# # 		print(s, m, n)
			
# 	t= time.time() - overall_time
# 	hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
# 	print("Total Time :", "\t%dh%dm%ds (%fs)" %(hours,minutes,seconds, t )  )

# 	print("saving results ... ")
# 	# diconame = os.path.join('./', outnpz)
# # 	diconame = '%s.%s' % (args.outnpz, 'npz')
# 	np.savez(args.outnpz, 
# 		OptNbSample_ns=OptNbSample_ns, 
# 		OptNbEpoch_ns=OptNbEpoch_ns, 
# 		OptMseTrain_ns = OptMseTrain_ns,
# 		OptMseValid_ns=OptMseValid_ns,
# 		TensorMseTrain_ns=TensorMseTrain_ns,
# 		TensorMseValid_ns=TensorMseValid_ns,
# 		OptNbSample_s=OptNbSample_s,
# 		OptNbEpoch_s=OptNbEpoch_s, 
# 		OptAccTrain_s=OptAccTrain_s, 
# 		OptAceTrain_s=OptAceTrain_s, 
# 		OptMseValid_s=OptMseValid_s, 
# 		OptAccValid_s = OptAccValid_s, 
# 		OptAceValid_s = OptAceValid_s, 
# 		TensorMseValid_s = TensorMseValid_s,
# 		TensorAceTrain_s = TensorAceTrain_s,
# 		TensorAceValid_s = TensorAceValid_s,
# 		TensorAccValid_s = TensorAccValid_s,
# 		ArrayAccTest = ArrayAccTest, 
# 		ArrayAceTest = ArrayAceTest,  
# 		ArrayMseTest = ArrayMseTest)

# 	write_model_data(network_enc, 'network_enc')
# 	write_model_data(network_class, 'network_class')
	

if __name__ == "__main__":
	print("guided fully-connected convolutional autoencoder")
	main()
	

