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
	
# 	network = lasagne.layers.MaxPool2DLayer( network, pool_size=(2,2) )
# 	print(lasagne.layers.get_output_shape(network))
	
# 	l_fcenc = lasagne.layers.DenseLayer(l_pool2, num_units=512, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
# 	l_fc = lasagne.layers.DenseLayer(l_fcenc, num_units=256, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	l_le = lasagne.layers.Conv2DLayer( network, num_filters=16, filter_size=(1,1), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform())
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
	
# 	l_mlp = lasagne.layers.DenseLayer(l_le, num_units=256, nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.GlorotUniform())
	l_outclass = lasagne.layers.DenseLayer(l_le, num_units=10, nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.GlorotUniform())
	
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

def main( num_epochs=10, num_exp=10, prop_valid=20 ):

	print("Set network")
	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')
	class_var = T.ivector('classes')
	
	network_enc, network_class = build_lae(input_var)
	
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
	mce_class = lasagne.objectives.aggregate(ce_class)
	params_class = lasagne.layers.get_all_params(network_class, trainable=True)
	updates_class = lasagne.updates.nesterov_momentum(mce_class, params_class, learning_rate=0.1, momentum=0.9)
	
	train_fn_class = theano.function([input_var, class_var], mce_class, updates=updates_class)
	
	test_class_prediction = lasagne.layers.get_output(network_class, deterministic=True)
	test_class_ce = lasagne.objectives.categorical_crossentropy(test_class_prediction, class_var)
	test_class_mce = lasagne.objectives.aggregate(test_class_ce)
	test_class_acc = T.mean( T.eq( T.argmax( test_class_prediction, axis=1 ), class_var ), dtype=theano.config.floatX )
	
	test_enc_reconstruction = lasagne.layers.get_output(network_enc, deterministic=True)
	test_enc_se = lasagne.objectives.squared_error(test_enc_reconstruction, target_var)
	test_enc_mse = lasagne.objectives.aggregate(test_enc_se)
	
	val_fn_class = theano.function([input_var, class_var], [test_class_mce, test_class_acc] )
	val_fn_enc = theano.function([input_var, target_var], test_enc_mse)
	
	
	# mnist dataset
	print("Loading mnist data...")
	X_train, y_train, X_test, y_test = load_dataset_mnist()
	
	overall_time = time.time()
	
	seqm = np.arange(100,-10,-10)
# 	m = 0
	seqn = np.arange(50)
	
	ListOfListAccTest = np.zeros( [len(seqm), len(seqn)] ) 
	ListOfListLossTest = np.zeros( [len(seqm), len(seqn)] ) 
	ListOfListMseTest = np.zeros ( [len(seqm), len(seqn)] )
	ListOfListOfListMseValid = np.zeros( [len(seqm), len(seqn), num_epochs] )
	ListOfListOfListLossValid = np.zeros( [len(seqm), len(seqn), num_epochs] )
	ListOfListOfListAccValid = np.zeros( [len(seqm), len(seqn), num_epochs] )
	ListOfListOfListMseTrain = np.zeros( [len(seqm), len(seqn), num_epochs] )
	ListOfListOfListLossTrain = np.zeros( [len(seqm), len(seqn), num_epochs] )
	ListOfListOfListAccTrain = np.zeros( [len(seqm), len(seqn), num_epochs] )
	
	
	for m in seqm/10 :
		prop_train_s = m * 10
		print("learning supervision rate", 100*s,"%" )
		for n in seqn:
			
			print("experiment:", n, "/", len(seqn))
			T_ind = np.arange(len(y_train))
			np.random.shuffle(T_ind)
			X_train = X_train[T_ind]
			y_train = y_train[T_ind]
			
			if prop_valid <= 0 or prop_valid>=100 :
				print("WARNING: validation/Training proportion cannot be 0% or 100% : setting default 20%....")
				prop_valid=20
			
			nb_train_s = np.floor( ( prop_train_s ) * len(X_train) ).astype(int)  # part used for the autoencoder (the rest for training the classifier)
			nb_valid_s = np.floor( (prop_valid/100)* len(X_train_s) ).astype(int)
			nb_valid_ns = np.floor( (prop_valid/100)* len(X_train_ns) ).astype(int)
			
			if nb_train_s !=0:
				# supervised / non-supervised split
				X_train_ns, X_train_s = X_train[:-nb_train_s], X_train[-nb_train_s:]
				y_train_ns, y_train_s = y_train[:-nb_train_s, y_train[-nb_train_s:]
				# train/validation split
				X_train_s, X_val_s = X_train_s[:-nb_valid_s], X_train_s[-nb_valid_s:]
				y_train_s, y_val_s = y_train_s[:-nb_valid_s], y_train_s[-nb_valid_s:]
				X_train_ns, X_val_ns = X_train_ns[:-nb_valid_ns], X_train_s[-nb_valid_s:]
				y_train_ns, y_val_ns = y_train_ns[:-nb_valid_ns], y_train_s[-nb_valid_ns:]
			elif nb_train_s == 0: # if supervision Rate 100% -> p = 0 -> the selection of indices [-0:] and [:-0] are permuted
				X_train_s, X_train_ns = X_train[:-nb_train_s], X_train[-nb_train_s:]
				y_train_s, y_train_ns = y_train[:-nb_train_s, y_train[-nb_train_s:]
				X_val_s, X_train_s = X_train_s[:-nb_valid_s], X_train_s[-nb_valid_s:]
				y_val_s, y_train_s = y_train_s[:-nb_valid_s], y_train_s[-nb_valid_s:]
				X_val_ns, X_train_ns = X_train_ns[:-nb_valid_ns], X_train_s[-nb_valid_s:]
				y_val_ns, y_train_ns = y_train_ns[:-nb_valid_ns], y_train_s[-nb_valid_ns:]
			print("nb images non-supervised Train/Val:", len(X_train_ns), "/", len(X_val_ns))
			print("nb images supervised Train/Val:", len(X_train_s), "/", len(X_val_s) )
			
			
			
			
# 			print(len(X_train_class), len(X_train_enc))
			
# 			print("Starting training...")
			# iteration over epochs:
			training_time = time.time()
			MseTrain_lowest = sys.float_info.max
			params_nn_ns_best = network_enc
			for e_ns in range(num_epochs):
				
				train_err = 0
				train_acc_class = 0
				train_err_enc = 0
				train_err_class = 0
				train_batches_enc = 0
				train_batches_class = 0
				start_time = time.time()
				
				### shuffle indices of train data
				
				ind_train_s = np.arange( len(y_train_s) )
				np.random.shuffle( ind_train_s )
				X_train_s = X_train_s[ ind_train_s ]
				y_train_s = y_train_s[ ind_train_s ]
				
				ind_train_ns = np.arange( len(y_train_ns) )
				np.random.shuffle( ind_train_ns )
				X_train_ns = X_train_ns[ ind_train_ns ]
				y_train_ns = y_train_ns[ ind_train_ns ]
				
				
				#### batch TRAIN ENCODER ####
				for batch in iterate_minibatches(X_train_ns, X_train_ns, y_train_ns, 500, shuffle=True):
					inputs, targets, classes = batch
					train_err_enc += train_fn_enc( inputs, targets )
					train_batches_enc += 1
				
				ListOfListOfListMseTrain[m][n][e_ns] = MseTrain
				ListOfListOfListLossTrain[m][n][e_ns] = LossTrain
				ListOfListOfListAccTrain[m][n][e_ns] = AccTrain
				print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
				print("\t training recons MSE:\t\t{:.6f} ".format( MseTrain ) )
				print("\t training class loss:\t\t{:.6f} ".format( LossTrain ) )
				print("\t training class acc:\t\t{:.2f} %".format( 100*( AccTrain ) ) )		
				
				
				for batch in iterate_minibatches(X_val, X_val, y_val, 500, shuffle=True):
					inputs, targets, classes = batch
					val_err_enc += val_fn_enc(inputs, targets)
		# 			val_err_class += val_fn_class(inputs, classes)
					err, acc= val_fn_class(inputs, classes)
					val_err_class += err
					val_acc_class += acc
					val_batches += 1
				# print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
# 				print("")
				print("\t validation recons loss:\t{:.6f}".format(val_err_enc / val_batches) )
				print("\t validation class loss:\t\t{:.6f}".format(val_err_class / val_batches) )
				print("\t validation class acc:\t\t{:.2f} %".format( 100*(val_acc_class / val_batches) ) )		
				
				ListOfListOfListMseValid[m][n][epoch] = val_err_enc / val_batches
				ListOfListOfListLossValid[m][n][epoch] = val_err_class / val_batches
				ListOfListOfListAccValid[m][n][epoch] = val_acc_class / val_batches
	
				if train_batches_enc != 0: 
					MseTrain = (train_err_enc / train_batches_enc)
				else: 
					MseTrain = 0
				
				print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
				print("\t training recons MSE:\t\t{:.6f} ".format( MseTrain ) )
				if MseTrain < MseTrain_lowest:
					MseTrain_lowest = MseTrain
					params_nn_ns_best = lasagne.layers.get_all_param_values(network_enc)
				
				
			for e_ns in range(num_epochs):
				
				#### batch TRAIN CLASSIFIER ####
				for batch in iterate_minibatches(X_train_class, X_train_class, y_train_class, 500, shuffle=True):
					inputs, targets, classes = batch
					train_err_class += train_fn_class(inputs, classes)
					err, acc= val_fn_class(inputs, classes)
					train_acc_class += acc
					train_batches_class += 1
		# 			print(" minibatch",train_batches, "of epoch", epoch + 1, ":", "\t{:.3f}s".format( time.time() - start_time) )
		# 			print("\t training loss class:\t{:.6f} ".format(train_err_class / train_batches) )
		# 			print("\t training accuracy:\t{:.2f} %".format( 100*(train_acc / train_batches) ) )		
			
					# print("		training loss enc =", train_err_enc / train_batches) 
					# print("		training loss class =", train_err_class / train_batches) 
					
				# simple verification if supervisionRate = 0% or 100 %
				if train_batches_class != 0:
					LossTrain = (train_err_class / train_batches_class)
					AccTrain = (train_acc_class / train_batches_class)
				else:
					LossTrain = 0
					AccTrain = 0
				# save training results
								
				
				# print("		training loss enc =", train_err_enc / train_batches) 
				# print("		training loss class =", train_err_class / train_batches) 
				# And a full pass over the validation data:
				val_err = 0
				val_err_enc = 0
				val_err_class = 0
				val_acc_class = 0
				val_batches = 0
				for batch in iterate_minibatches(X_val, X_val, y_val, 500, shuffle=True):
					inputs, targets, classes = batch
					val_err_enc += val_fn_enc(inputs, targets)
		# 			val_err_class += val_fn_class(inputs, classes)
					err, acc= val_fn_class(inputs, classes)
					val_err_class += err
					val_acc_class += acc
					val_batches += 1
				# print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - start_time))
# 				print("")
				print("\t validation recons loss:\t{:.6f}".format(val_err_enc / val_batches) )
				print("\t validation class loss:\t\t{:.6f}".format(val_err_class / val_batches) )
				print("\t validation class acc:\t\t{:.2f} %".format( 100*(val_acc_class / val_batches) ) )		
				
				ListOfListOfListMseValid[m][n][epoch] = val_err_enc / val_batches
				ListOfListOfListLossValid[m][n][epoch] = val_err_class / val_batches
				ListOfListOfListAccValid[m][n][epoch] = val_acc_class / val_batches
			
			test_err = 0
			test_err_enc = 0
			test_err_class = 0
			test_acc_class = 0
			test_batches = 0
			for batch in iterate_minibatches(X_test, X_test, y_test, 500, shuffle=True):
				inputs, targets, classes = batch
				test_err_enc += val_fn_enc(inputs, targets)
		# 		test_err_class += val_fn_class(inputs, classes)
				err, acc= val_fn_class(inputs, classes)
				test_err_class += err
				test_acc_class += acc
				test_batches += 1
# 			print("Epoch :", epoch + 1, "/", num_epochs, "\t{:.3f}s".format( time.time() - training_time))
			print("Test Results | supervision rate ", 100*s, "% | experiment ", n, "/", len(seqn) )
			print("\t test recons loss:\t\t{:.6f}".format(test_err_enc / test_batches) )
			print("\t test class loss:\t\t{:.6f}".format(test_err_class / test_batches) )
			print("\t test class acc:\t\t{:.2f} %".format( 100*(test_acc_class / test_batches) ) )		
			ListOfListAccTest[m][n] =  (test_acc_class / test_batches) 
			ListOfListLossTest[m][n] =  (test_err_class / test_batches) 
			ListOfListMseTest[m][n] =  (test_err_enc / test_batches) 
		
# 		print(s, m, n)
			
		
	print("Total Time :", "\t{:.3f}s".format( time.time() - overall_time))
	print("saving results ... ")
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
	diconame = os.path.join('./', 'results')
	diconame = '%s.%s' % (diconame, 'npz')
	np.savez(diconame, 
		TestAcc = ListOfListAccTest, 
		TestLoss = ListOfListLossTest, 
		TestMse = ListOfListMseTest , 
		ValEpochAcc = ListOfListOfListAccValid ,
		ValEpochLoss = ListOfListOfListLossValid , 
		ValEpochMse = ListOfListOfListMseValid ,
		TrainEpochAcc = ListOfListOfListAccTrain ,
		TrainEpochLoss = ListOfListOfListLossTrain ,
		TrainEpochMse = ListOfListOfListMseTrain )

# 	write_model_data(network_enc, 'network_enc')
# 	write_model_data(network_class, 'network_class')
	

if __name__ == "__main__":
	print("guided fully-connected convolutional autoencoder")
	main()
	

