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
	parser.add_argument("-s", dest="seq", help="list of number", nargs="+", type=int)

	args = parser.parse_args()
	lala = args.seq
	print(lala)
	fnp = read_npz(args.FILENPZ)
	nameHeader = ["OptNbSample_ns", "OptMseTrain_ns", "OptMseValid_ns","TensorMseTrain_ns","TensorMseValid_ns","OptNbSample_s","OptAccTrain_s","OptAceTrain_s","OptMseValid_s","OptAccValid_s","OptAceValid_s","TensorMseValid_s","TensorAceTrain_s","TensorAceValid_s","TensorAccValid_s","ArrayAccTest","ArrayAceTest","ArrayMseTest"]
	AccTest = getattr( fnp.f, 'ArrayAccTest').T
	AceTest = getattr( fnp.f, 'ArrayAceTest').T
	MseTest = getattr( fnp.f, 'ArrayMseTest').T

	fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharey=False, sharex=True)
	axes[0].set_ylim(0.8, 1.05)
	#axes[1].set_ylim(0.05, 1.05)
	#axes[2].set_ylim(0.05, 0.5)

	dfAccTest = pd.DataFrame( AccTest )
	dfAccTest.plot.box(ax=axes[0], title='Accuracy of classification TEST depending on Supervisation Rate (Supervised/Unsupervised)')
	print(dfAccTest)
	dfAceTest = pd.DataFrame( AceTest )
	dfAceTest.plot.box(ax=axes[1], title='Average Cross Entropy on classification TEST depending on Supervisation Rate (Supervised/Unsupervised)')
	dfMseTest = pd.DataFrame( MseTest )
	dfMseTest.plot.box(ax=axes[2], title='Mean Square Error on reconstruction TEST depending on Supervisation Rate (Supervised/Unsupervised)')
	plt.show()


if __name__ == "__main__":
	print("open and display [dlrecital] experiments results")
	main()
