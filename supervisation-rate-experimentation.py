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
# 	X_train, X_val = X_train[:-10000], X_train[-10000:]
# 	y_train, y_val = y_train[:-10000], y_train[-10000:]
	
	
	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_test, y_test


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
	
	l_le = lasagne.layers.Conv2DLayer( network, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform() )
	print(lasagne.layers.get_output_shape(l_le))
	
	network = lasagne.layers.Deconv2DLayer( l_le, num_filters=32, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	print(lasagne.layers.get_output_shape(network))
	
# 	l_fcdec = lasagne.layers.DenseLayer(l_fc, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
# 	l_reshpfcdec = lasagne.layers.ReshapeLayer( l_fcdec, shape=(-1, 32, 4, 4) )
# 	l_upscale1 = lasagne.layers.Upscale2DLayer(l_reshpfcdec, scale_factor=2, mode='repeat')

# 	network = lasagne.layers.Upscale2DLayer(network, scale_factor=2, mode='repeat')
# 	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Deconv2DLayer( network, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Upscale2DLayer(network, scale_factor=2, mode='repeat')
	print(lasagne.layers.get_output_shape(network))
	
	network = lasagne.layers.Deconv2DLayer(network, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform() )
	print(lasagne.layers.get_output_shape(network))
	
	l_out = lasagne.layers.FeaturePoolLayer(network, 32, pool_function=theano.tensor.max )
	
	l_mlp = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_le, p=0.5), num_units=256)
	l_outclass = lasagne.layers.DenseLayer(l_mlp, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
	
	print("output class:", lasagne.layers.get_output_shape(l_outclass))
	
	print("output reconstruction:",lasagne.layers.get_output_shape(l_out))
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

def main( num_epochs=100, num_exp=10, prop_valid=20, size_minibatch = 1000 ):

	print("Set network")
	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')
	class_var = T.ivector('classes')
	
	network_enc, network_class = build_lae(input_var)
	params_init_network_enc = lasagne.layers.get_all_param_values(network_enc)
	params_init_network_class = lasagne.layers.get_all_param_values(network_class)

	print("number of params for enc",lasagne.layers.count_params(network_enc))
	print("number params for class ",lasagne.layers.count_params(network_class))
	
	# definition of what is "train" for encoder
	reconstruction_enc = lasagne.layers.get_output(network_enc)
	se_enc = lasagne.objectives.squared_error(reconstruction_enc, target_var)
	mse_enc = lasagne.objectives.aggregate(se_enc)
	params_enc = lasagne.layers.get_all_params(network_enc, trainable=True)
	updates_enc = lasagne.updates.nesterov_momentum(mse_enc, params_enc, learning_rate=0.1, momentum=0.9)
	
	train_fn_enc = theano.function([input_var, target_var], mse_enc, updates=updates_enc)
	
	
	# definition of what is "train" for classifier 
	reconstruction_enc = lasagne.layers.get_output(network_enc)
	prediction_class = lasagne.layers.get_output(network_class)
	ce_class = lasagne.objectives.categorical_crossentropy(prediction_class, class_var)
	ace_class = lasagne.objectives.aggregate(ce_class)
	params_class = lasagne.layers.get_all_params(network_class, trainable=True)
	updates_class = lasagne.updates.nesterov_momentum(ace_class, params_class, learning_rate=0.1, momentum=0.9)
	
	train_fn_class = theano.function([input_var, class_var], ace_class, updates=updates_class)
	
	test_class_prediction = lasagne.layers.get_output(network_class, deterministic=True)
	test_class_ce = lasagne.objectives.categorical_crossentropy(test_class_prediction, class_var)
	test_class_ace = lasagne.objectives.aggregate(test_class_ce)
	test_class_acc = T.mean( T.eq( T.argmax( test_class_prediction, axis=1 ), class_var ), dtype=theano.config.floatX )
	
	test_enc_reconstruction = lasagne.layers.get_output(network_enc, deterministic=True)
	test_enc_se = lasagne.objectives.squared_error(test_enc_reconstruction, target_var)
	test_enc_mse = lasagne.objectives.aggregate(test_enc_se)
	
	val_fn_class = theano.function([input_var, class_var], [test_class_ace, test_class_acc] )
	val_fn_enc = theano.function([input_var, target_var], test_enc_mse)
	
	
	overall_time = time.time()
	# mnist dataset
	print("Loading mnist data...")
	X_train, y_train, X_test, y_test = load_dataset_mnist()
	seqm = np.arange(100,-10, -10)
	seqn = np.arange(num_exp)
	
	
# 	m = 0

	### results for NS training ###
	OptNbSample_ns = np.zeros ( [len(seqm), len(seqn)] )
	OptMseTrain_ns = np.zeros ( [len(seqm), len(seqn)] )
	OptMseValid_ns = np.zeros ( [len(seqm), len(seqn)] )
	TensorMseTrain_ns = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorMseValid_ns = np.zeros( [len(seqm), len(seqn), num_epochs] )

	### results for S training ###
	OptNbSample_s = np.zeros ( [len(seqm), len(seqn)] )
	OptAccTrain_s = np.zeros( [len(seqm), len(seqn)] ) 
	OptAceTrain_s = np.zeros( [len(seqm), len(seqn)] ) 
	OptMseValid_s = np.zeros( [len(seqm), len(seqn)] ) 
	OptAccValid_s = np.zeros( [len(seqm), len(seqn)] ) 
	OptAceValid_s = np.zeros( [len(seqm), len(seqn)] ) 
# 	TensorMseTrain_s = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorMseValid_s = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAceTrain_s = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAceValid_s = np.zeros( [len(seqm), len(seqn), num_epochs] )
# 	TensorAccTrain_s = np.zeros( [len(seqm), len(seqn), num_epochs] )
	TensorAccValid_s = np.zeros( [len(seqm), len(seqn), num_epochs] )

	### results Test ###
	ArrayAccTest = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayAceTest = np.zeros( [len(seqm), len(seqn)] ) 
	ArrayMseTest = np.zeros ( [len(seqm), len(seqn)] )
	
	for m in np.arange(len(seqm)) :
		
		prop_train_s = seqm[m] 
		print("learning supervision rate", prop_train_s,"%" )
		for n in seqn:
			print("re-initialize network parameters ... ")
			lasagne.layers.set_all_param_values( network_enc, params_init_network_enc )
			lasagne.layers.set_all_param_values( network_class, params_init_network_class )
			
			print("experiment:", n+1, "/", len(seqn))
			T_ind = np.arange(len(y_train))
			np.random.shuffle(T_ind)
			X_train = X_train[T_ind]
			y_train = y_train[T_ind]
			
			if prop_valid <= 0 or prop_valid>=100 :
				print("WARNING: validation/Training proportion cannot be 0% or 100% : setting default 20%....")
				prop_valid=20
			
			nb_train_s = np.floor( ( prop_train_s/100 ) * len(X_train) ).astype(int)  # part used for the supervised learning
			nb_train_ns = np.floor( (1-(prop_train_s/100)) * len(X_train) ).astype(int)  # part used for the autoencoder (the rest for training the classifier)
			if nb_train_s !=0:
				# supervised / non-supervised split
				X_train_ns, X_train_s = X_train[:-nb_train_s], X_train[-nb_train_s:]
				y_train_ns, y_train_s = y_train[:-nb_train_s], y_train[-nb_train_s:]
			elif nb_train_s == 0: # if supervision Rate 100% -> p = 0 -> the selection of indices [-0:] and [:-0] are permuted
				X_train_s, X_train_ns = X_train[:-nb_train_s], X_train[-nb_train_s:]
				y_train_s, y_train_ns = y_train[:-nb_train_s], y_train[-nb_train_s:]
			
			nb_valid_s = np.floor( (prop_valid/100)* len(X_train_s) ).astype(int)
			nb_valid_ns = np.floor( (prop_valid/100)* len(X_train_ns) ).astype(int)
# 			print("nb_valid_s", nb_valid_s )
# 			print("nb_valid_ns", nb_valid_ns )
			if nb_valid_s !=0:
				# train/validation split
				X_train_s, X_val_s = X_train_s[:-nb_valid_s], X_train_s[-nb_valid_s:]
				y_train_s, y_val_s = y_train_s[:-nb_valid_s], y_train_s[-nb_valid_s:]
			elif nb_valid_s == 0: # if supervision Rate 100% -> p = 0 -> the selection of indices [-0:] and [:-0] are permuted
				X_val_s, X_train_s = X_train_s[:-nb_valid_s], X_train_s[-nb_valid_s:]
				y_val_s, y_train_s = y_train_s[:-nb_valid_s], y_train_s[-nb_valid_s:]
			print("number images for supervised learning (train/val):", nb_train_s , "(", len(X_train_s), "/", len(X_val_s),")")
# 			print("number images for non-supervised learning (train/val):", len(X_train_ns) )
# 			print("Split supervised Train/Val:", len(X_train_s), "/", len(X_val_s) )
		
			if nb_valid_ns !=0:
				# train/validation split
				X_train_ns, X_val_ns = X_train_ns[:-nb_valid_ns], X_train_ns[-nb_valid_ns:]
				y_train_ns, y_val_ns = y_train_ns[:-nb_valid_ns], y_train_ns[-nb_valid_ns:]
			elif nb_valid_ns == 0: # if supervision Rate 100% -> p = 0 -> the selection of indices [-0:] and [:-0] are permuted
				X_val_ns, X_train_ns = X_train_ns[:-nb_valid_ns], X_train_ns[-nb_valid_ns:]
				y_val_ns, y_train_ns = y_train_ns[:-nb_valid_ns], y_train_ns[-nb_valid_ns:]
			print("number images for non-supervised learning (train/val):", nb_train_ns , "(", len(X_train_ns), "/", len(X_val_ns),")")
# 			print("Split non-supervised Train/Val:", len(X_train_ns), "/", len(X_val_ns))
			
			
# 			print(len(X_train_class), len(X_train_enc))
			
# 			print("Starting training...")
			# iteration over epochs:
			training_time = time.time()
			MseTrain_lowest = sys.float_info.max
			MseVal_lowest = sys.float_info.max
			best_nbsample = 0
			params_nn_ns_best = lasagne.layers.get_all_param_values(network_enc)
			
			for e_ns in range(num_epochs):
				
				train_mse = 0
				train_batches = 0
				val_mse = 0
				val_batches = 0
				start_time = time.time()
				
				### shuffle indices of train/valid data
				
				ind_train_ns = np.arange( len(y_train_ns) )
				np.random.shuffle( ind_train_ns )
				X_train_ns = X_train_ns[ ind_train_ns ]
				y_train_ns = y_train_ns[ ind_train_ns ]
				
				
				#### batch TRAIN ENCODER ####
				for batch in iterate_minibatches(X_train_ns, X_train_ns, y_train_ns, size_minibatch, shuffle=True):
					inputs, targets, classes = batch
					train_mse += train_fn_enc( inputs, targets )
					train_batches += 1
				
				MseTrain = 0
				if train_batches != 0: 
					MseTrain = (train_mse / train_batches)

				#### batch VALID ENCODER ####
				for batch in iterate_minibatches(X_val_ns, X_val_ns, y_val_ns, size_minibatch, shuffle=True):
					inputs, targets, classes = batch
					val_mse += val_fn_enc(inputs, targets)
					val_batches += 1
				
				MseVal = 0
				if val_batches != 0: 
					MseVal = (val_mse / val_batches)
				t = time.time() - overall_time
# 				hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, ((t - 3600*(t//3600)) - 60*(t - (3600*(t//3600)))//60)
				hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
				print("-----UnSupervised-----")
				print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
				print("")	
				print("Epoch: ", e_ns + 1, "/", num_epochs, "\tn:%d/%d" % (n+1,len(seqn)), "\tt: {:.3f}s".format( time.time() - start_time), "\ts: %d" %(prop_train_s), "%")
				print("\t training recons MSE:\t{:.6f} ".format( MseTrain ) )
				print("\t validation recons MSE:\t{:.6f}".format( MseVal ) )
				print("")	
				
				TensorMseTrain_ns[m][n][e_ns] = MseTrain
				TensorMseValid_ns[m][n][e_ns] = MseVal
				
				if MseVal < MseVal_lowest:
					MseVal_lowest = MseVal
					OptMseTrain_ns[m][n] = MseVal
					OptMseTrain_ns[m][n] = MseTrain
					OptNbSample_ns[m][n] = e_ns * len(X_train_ns)
					params_nn_ns_best = lasagne.layers.get_all_param_values(network_enc)
					params_nn_s_best = lasagne.layers.get_all_param_values(network_class)
				
# 				ListOfListOfListLossTrain[m][n][e_ns] = LossTrain
# 				ListOfListOfListAccTrain[m][n][e_ns] = AccTrain
# 				print("\t training class loss:\t\t{:.6f} ".format( LossTrain ) )
# 				print("\t training class acc:\t\t{:.2f} %".format( 100*( AccTrain ) ) )		
				
				
# 				print("")
# 				print("\t validation class loss:\t\t{:.6f}".format(val_err_class / val_batches) )
# 				print("\t validation class acc:\t\t{:.2f} %".format( 100*(val_acc_class / val_batches) ) )		
				
# 				ListOfListOfListMseValid[m][n][epoch] = val_err_enc / val_batches
# 				ListOfListOfListLossValid[m][n][epoch] = val_err_class / val_batches
# 				ListOfListOfListAccValid[m][n][epoch] = val_acc_class / val_batches
			
				
			lasagne.layers.set_all_param_values( network_enc, params_nn_ns_best )

			AceTrain_lowest = sys.float_info.max
			AceVal_lowest = sys.float_info.max
			best_nbsample = 0
			params_nn_s_best = lasagne.layers.get_all_param_values(network_class)
			
			for e_s in range(num_epochs):
				
				train_ace= 0
				val_ace = 0
				val_acc = 0
				val_mse = 0
				train_batches = 0
				val_batches = 0
				start_time = time.time()
				
				### shuffle indices of train/valid data
				
				ind_train_s = np.arange( len(y_train_s) )
				np.random.shuffle( ind_train_s )
				X_train_s = X_train_s[ ind_train_s ]
				y_train_s = y_train_s[ ind_train_s ]
				
				
				#### batch TRAIN CLASSIFIER ####
				for batch in iterate_minibatches(X_train_s, X_train_s, y_train_s, size_minibatch, shuffle=True):
					inputs, targets, classes = batch
					train_ace += train_fn_class( inputs, classes )
					train_batches += 1
				
				if train_batches != 0: 
					AceTrain = (train_ace / train_batches)
				else: 
					AceTrain = 0

				#### batch VALID CLASSIFIER ####
				for batch in iterate_minibatches(X_val_s, X_val_s, y_val_s, size_minibatch, shuffle=True):
					inputs, targets, classes = batch
					ace, acc = val_fn_class( inputs, classes )
					val_ace += ace
					val_acc += acc
					val_mse += val_fn_enc(inputs, targets)
					val_batches += 1
				
				if val_batches != 0: 
					MseVal = (val_mse / val_batches)
					AceVal = (val_ace / val_batches)
					AccVal = (val_acc / val_batches)
				else: 
					MseVal = 0
					AceVal = 0
					AccVal = 0
				
				t = time.time() - overall_time
# 				hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, ((t - 3600*(t//3600)) - 60*(t - (3600*(t//3600)))//60)
				hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
				print("-----Supervised-----")
				print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
				print("")
				print("Epoch: ", e_s + 1, "/", num_epochs, "\tn:%d/%d" % (n+1,len(seqn)), "\tt: {:.3f}s".format( time.time() - start_time), "\ts: %d" %(prop_train_s), "%")
# 				print("Epoch :", e_s + 1, "/", num_epochs, "\tt: {:.3f}s".format( time.time() - start_time), "\tSR: {:1f}".format(prop_train_s), "%" )
# 				print("Epoch :", e_s + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
				print("\t training class ACE:\t{:.6f} ".format( AceTrain ) )
				print("\t validation class ACE:\t{:.6f}".format( AceVal ) )
				print("\t validation class MSE:\t{:.6f}".format( AceVal ) )
				print("\t validation class ACC:\t{:.6f}".format( AceVal ) )
				print("")
				TensorAceTrain_s[m][n][e_s] = AceTrain
				TensorMseValid_s[m][n][e_s] = MseVal
				TensorAceValid_s[m][n][e_s] = AceVal
				TensorAccValid_s[m][n][e_s] = AccVal
				
				if AceVal < AceVal_lowest:
					AceVal_lowest = AceVal
					OptAceTrain_s[m][n] = AceTrain
					OptAceValid_s[m][n] = AceVal
					OptMseValid_s[m][n] = MseVal
					OptAccValid_s[m][n] = AccVal
					OptNbSample_s[m][n] = e_s * len(X_train_s)
					params_nn_s_best = lasagne.layers.get_all_param_values(network_class)
					
			
			lasagne.layers.set_all_param_values( network_enc, params_nn_ns_best )
			lasagne.layers.set_all_param_values( network_class, params_nn_s_best )
			test_ace = 0
			test_acc = 0
			test_mse = 0
			test_batches = 0
			for batch in iterate_minibatches(X_test, X_test, y_test, size_minibatch, shuffle=True):
				inputs, targets, classes = batch
				ace, acc = val_fn_class( inputs, classes )
				test_ace += ace
				test_acc += acc
				test_mse += val_fn_enc(inputs, targets)
				test_batches += 1
			
			MseTest = (test_mse / test_batches)
			AceTest = (test_ace / test_batches)
			AccTest = (test_acc / test_batches)
			ArrayAccTest[m][n] = AccTest
			ArrayAceTest[m][n] = AceTest
			ArrayMseTest[m][n] = MseTest
				
# 			print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - training_time))
			print("Test Results | supervision rate ", prop_train_s, "% | experiment ", n, "/", len(seqn) )
			print("\t test recons MSE:\t\t{:.6f}".format( MseTest) )
			print("\t test class ACE:\t\t{:.6f}".format( AceTest) )
			print("\t test class ACC:\t\t{:.2f} %".format( 100*(AccTest) ) )
			
# 		print(s, m, n)
			
	t= time.time() - overall_time
	hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
	print("Total Time :", "\t%dh%dm%ds" %(hours,minutes,seconds) )
# 	dico = {}
# 	dico['TestAcc'] = ListOfListAccTest
# 	dico['TestLoss'] = ListOfListLossTest
# 	dico['TestMse'] = ListOfListMseTest
# 	dico['ValEpochAcc'] = ListOfListOfListAccValid
# 	dico['ValEpochLoss'] = ListOfListOfListLossValid
# 	dico['ValEpochMse'] = ListOfListOfListMseValid
# 	dico['TrainEpochAcc'] = ListOfListOfListAccTrain
# 	dico['TrainEpochLoss'] = ListOfListOfListLossTrain
# 	dico['TrainEpochMse'] = ListOfListOfListMseTrain
	print("saving results ... ")
	diconame = os.path.join('./', 'exp-supervision-rate')
	diconame = '%s.%s' % (diconame, 'npz')
	np.savez(diconame, 
		OptNbSample_ns=OptNbSample_ns, 
		OptMseTrain_ns = OptMseTrain_ns,
		OptMseValid_ns=OptMseValid_ns,
		TensorMseTrain_ns=TensorMseTrain_ns,
		TensorMseValid_ns=TensorMseValid_ns,
		OptNbSample_s=OptNbSample_s,
		OptAccTrain_s=OptAccTrain_s, 
		OptAceTrain_s=OptAceTrain_s, 
		OptMseValid_s=OptMseValid_s, 
		OptAccValid_s = OptAccValid_s, 
		OptAceValid_s = OptAceValid_s, 
# 		TensorMseTrain_s,
		TensorMseValid_s = TensorMseValid_s,
		TensorAceTrain_s = TensorAceTrain_s,
		TensorAceValid_s = TensorAceValid_s,
# 		TensorAccTrain_s,
		TensorAccValid_s = TensorAccValid_s,
		ArrayAccTest = ArrayAccTest, 
		ArrayAceTest = ArrayAceTest,  
		ArrayMseTest = ArrayMseTest)
# 	write_model_data(network_enc, 'network_enc')
# 	write_model_data(network_class, 'network_class')
	

if __name__ == "__main__":
	print("guided fully-connected convolutional autoencoder")
	main()
	

