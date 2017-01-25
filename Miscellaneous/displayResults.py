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
	parser = argparse.ArgumentParser(description="open [file.npz] and convert it into csv", formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("FILENPZ", help="fileName of the input image")
	args = parser.parse_args()
	filename = args.FILENPZ
	fnp = read_npz(filename)
	print(fnp.keys())
	header = fnp.keys()
# 	header = ["OptNbSample_ns","OptMseTrain_ns","OptMseValid_ns","TensorMseTrain_ns","TensorMseValid_ns","OptNbSample_s","OptAccTrain_s","OptAceTrain_s","OptMseValid_s","OptAccValid_s","OptAceValid_s","TensorMseValid_s","TensorAceTrain_s","TensorAceValid_s","TensorAccValid_s","ArrayAccTest","ArrayAceTest","ArrayMseTest"]
	for h in header:
# 		arr_i = '%s%s' % ('arr_', i)
		k = getattr(fnp.f, h)
# 		print(fnp.keys())
		print(k.shape)

if __name__ == "__main__":
	print("open and display [dlrecital] experiments results")
	main()
	


