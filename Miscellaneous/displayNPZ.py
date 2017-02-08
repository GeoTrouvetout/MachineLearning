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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot  as pyplot
from ggplot import *

plt.style.use('ggplot')
########### FUNCTIONS DEF ################


def read_npz(filename):
	file = np.load(filename)
	return file


def main():
	parser = argparse.ArgumentParser(description="open [file.npz] and convert it into csv")
	parser.add_argument("FILENPZ", help="fileName of the input array")

	args = parser.parse_args()
	fnp = read_npz(args.FILENPZ)
	print(fnp.keys())
	nameHeader = ["OptNbSample_ns", "OptMseTrain_ns", "OptMseValid_ns","TensorMseTrain_ns","TensorMseValid_ns","OptNbSample_s","OptAccTrain_s","OptAceTrain_s","OptMseValid_s","OptAccValid_s","OptAceValid_s","TensorMseValid_s","TensorAceTrain_s","TensorAceValid_s","TensorAccValid_s","ArrayAccTest","ArrayAceTest","ArrayMseTest"]
	AccTestclass = getattr( fnp.f, 'ArrayAccTest_class').T
	AccTestcnn = getattr( fnp.f, 'ArrayAccTest_cnn').T
	AccTestmlp = getattr( fnp.f, 'ArrayAccTest_mlp').T
# 	MseTest = getattr( fnp.f, 'ArrayMseTest_class').T

	fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharey=True, sharex=True)
	axes[0].set_ylim(0.8, 1.05)
	#axes[1].set_ylim(0.05, 1.05)
	#axes[2].set_ylim(0.05, 0.5)
	labcols=[100,90,80,70,60,50,40,30,20,10,9,8,7,6,5,4,3,2,1,0]
	dfAccTestclass = pd.DataFrame( AccTestclass, columns=labcols )
	dfAccTestclass.plot.box(ax=axes[2], title='Classification accuracy of Local Feature AutoEncoder (LFAE) depending on Supervisation Rate (Supersised/Unsupervised)')
	print(dfAccTestclass)
	dfAccTestmlp = pd.DataFrame( AccTestmlp , columns=labcols )
	dfAccTestmlp.plot.box(ax=axes[0], title='Classification accuracy of MLP depending on number of training data (given by the Supervisation Rate)')
	dfAccTestcnn = pd.DataFrame( AccTestcnn , columns=labcols )
	dfAccTestcnn.plot.box(ax=axes[1], title='Classification accuracy of CNN depending on the number of training data used (given by the Supervisation Rate)')
	plt.show()


if __name__ == "__main__":
	print("open and display [dlrecital] experiments results")
	main()
