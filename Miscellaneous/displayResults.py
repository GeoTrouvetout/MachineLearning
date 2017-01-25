#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Copyright 2016 geo trouvetout

Created on Mon Jan 2017

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


def read_npz(filename):
	file = np.load(filename)
	return file


def main():
	parser = argparse.ArgumentParser(description="open [file.npz] and convert it into csv")
	parser.add_argument("FILENPZ", help="fileName of the input image")
	args = parser.parse_args()
	filename = args.FILENPZ
	fnp = read_npz(filename)
	# print(fnp.keys())
	header = fnp.keys()
	nameHeader = ["OptNbSample_ns", "OptMseTrain_ns", "OptMseValid_ns","TensorMseTrain_ns","TensorMseValid_ns","OptNbSample_s","OptAccTrain_s","OptAceTrain_s","OptMseValid_s","OptAccValid_s","OptAceValid_s","TensorMseValid_s","TensorAceTrain_s","TensorAceValid_s","TensorAccValid_s","ArrayAccTest","ArrayAceTest","ArrayMseTest"]
	nbMatrix = 0
	nbTensor = 0
	
	for i in np.arange(len(header)):
		arr_i = '%s%s' % ('arr_', i)
		k = getattr(fnp.f, arr_i)
# 		print(fnp.keys())
		print(nameHeader[i], k.shape)
		
		if len(k.shape) == 2:
			nbMatrix += 1
		if len(k.shape) == 3:
			nbTensor += 1


	print(nbMatrix)
	print(nbTensor)

	fig, axes = plt.subplots(nrows=3, 
		figsize=(6, 6), sharey=True)
	ArrayAccTest = getattr(fnp.f, 'arr_15').T
	axes[0].boxplot(ArrayAccTest)
	ArrayAceTest = getattr(fnp.f, 'arr_16')
	axes[1].boxplot(ArrayAceTest)
	ArrayMseTest = getattr(fnp.f, 'arr_17')
	axes[2].boxplot(ArrayMseTest)
	
	plt.show()
	# axes[1].boxplot(ArrayAccTest, labels=)
	# axes[1].boxplot(ArrayAceTest, labels=labels)
	# axes[3].boxplot(ArrayMseTest, labels=labels)
	# plt.show()
	# for h in header:
	# 	k = getattr(fnp.f, h)
	# 	if len(k.shape) == 2:
	# 		axes[0, 0].boxplot(k, labels=labels)

	# axes[0, 0].boxplot(k, labels=labels)
	# axes[0, 0].set_title('Default', fontsize=fs)
	# print("wow ... ;)")

if __name__ == "__main__":
	print("open and display [dlrecital] experiments results")
	main()
	


