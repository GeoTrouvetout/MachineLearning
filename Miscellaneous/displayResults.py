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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from ggplot import *

plt.style.use('ggplot')
########### FUNCTIONS DEF ################


def read_npz(filename):
	file = np.load(filename)
	return file


def main():
	parser = argparse.ArgumentParser(description="open [file.npz] and convert it into csv")
	parser.add_argument("FILENPZ", help="fileName of the input image")
	parser.add_argument("FILENPZ2", help="fileName of the input image")

	args = parser.parse_args()
	filename = args.FILENPZ
	fnp = read_npz(args.FILENPZ)
	fnp2 = read_npz(args.FILENPZ2)

	# pd.DataFrame(data=fnp.f[1:,1:],index=data[1:,0],columns=data[0,1:])
	# print(fnp.keys())

	header = fnp.keys()
	nameHeader = ["OptNbSample_ns", "OptMseTrain_ns", "OptMseValid_ns","TensorMseTrain_ns","TensorMseValid_ns","OptNbSample_s","OptAccTrain_s","OptAceTrain_s","OptMseValid_s","OptAccValid_s","OptAceValid_s","TensorMseValid_s","TensorAceTrain_s","TensorAceValid_s","TensorAccValid_s","ArrayAccTest","ArrayAceTest","ArrayMseTest"]
	nbMatrix = 0
	nbTensor = 0
	
	plt.figure(0)
	ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

	for h in header:
		# arr_i = '%s%s' % ('arr_', i)
		k = getattr(fnp.f, h)

# # 		print(fnp.keys())
		print(h, k.shape)
		
		if len(k.shape) == 2:
			nbMatrix += 1
			axes[]plot.box(ax=axes[0,0])

		if len(k.shape) == 3:
			nbTensor += 1
			# km = k.reshape(k.shape[0]*k.shape[1], k.shape[2])
			# for i in np.arange(k.shape[0]) 
			# 	k[i,:,:]
			# print(km.shape)


	print(nbMatrix)
	print(nbTensor)

	fig, axes = plt.subplots(nrows=, ncols=3, figsize=(6, 6), sharey=True)

	AccTest = getattr(fnp.f, 'ArrayAccTest').T
	AccTest_wu = getattr(fnp2.f, 'ArrayAccTest').T
	AceTest = getattr(fnp.f, 'ArrayAceTest').T
	AceTest_wu = getattr(fnp2.f, 'ArrayAceTest').T
	MseTest = getattr(fnp.f, 'ArrayMseTest').T
	MseTest_wu = getattr(fnp2.f, 'ArrayMseTest').T
	# fig, axes = plt.subplots(nrows=3, figsize=(6, 6), sharey=True)
	AccValid_s = getattr(fnp.f, 'OptAccValid_s').T
	AccValid_s_wu = getattr(fnp2.f, 'OptAccValid_s').T
	AceValid_s = getattr(fnp.f, 'OptAceValid_s').T
	AceValid_s_wu = getattr(fnp2.f, 'OptAceValid_s').T
	MseValid_s = getattr(fnp.f, 'OptMseValid_s').T
	MseValid_s_wu = getattr(fnp2.f, 'OptMseValid_s').T
	# for i in range(OptAccValid_s.shape[1]):
	# 

	plt.ylim(-.05, 1.05)

	dfAccValid_s=pd.DataFrame(AccValid_s)
	dfAccValid_s.plot.box(ax=axes[0,0])

	dfAceValid_s=pd.DataFrame(AceValid_s)
	dfAceValid_s.plot.box(ax=axes[1,0])

	dfMseValid_s=pd.DataFrame(MseValid_s)
	dfMseValid_s.plot.box(ax=axes[2,0])

	dfAccValid_s_wu=pd.DataFrame(AccValid_s_wu)
	dfAccValid_s_wu.plot.box(ax=axes[3,0])

	dfAceValid_s_wu=pd.DataFrame(AceValid_s_wu)
	dfAceValid_s_wu.plot.box(ax=axes[4,0])

	dfMseValid_s_wu=pd.DataFrame(MseValid_s_wu)
	dfMseValid_s_wu.plot.box(ax=axes[5,0])

	dfAccTest=pd.DataFrame(AccTest)
	dfAccTest.plot.box(ax=axes[0,2])

	dfAceTest=pd.DataFrame(AceTest)
	dfAceTest.plot.box(ax=axes[1,2])

	dfMseTest=pd.DataFrame(MseTest)
	dfMseTest.plot.box(ax=axes[2,2])

	dfAccTest_wu=pd.DataFrame(AccTest_wu)
	dfAccTest_wu.plot.box(ax=axes[3,2])

	dfAceTest_wu=pd.DataFrame(AceTest_wu)
	dfAceTest_wu.plot.box(ax=axes[4,2])

	dfMseTest_wu=pd.DataFrame(MseTest_wu)
	dfMseTest_wu.plot.box(ax=axes[5,2])


	n=1
	BestAccValid_s = np.median( getattr(fnp.f, 'TensorAccValid_s') , axis=n)
	BestAceValid_s = np.median( getattr(fnp.f, 'TensorAceValid_s') , axis=n)
	BestMseValid_s = np.median( getattr(fnp.f, 'TensorMseValid_s') , axis=n)
	
	BestAccValid_s_wu = np.median( getattr(fnp2.f, 'TensorAccValid_s') , axis=n)
	BestAceValid_s_wu = np.median( getattr(fnp2.f, 'TensorAceValid_s') , axis=n)
	BestMseValid_s_wu = np.median( getattr(fnp2.f, 'TensorMseValid_s') , axis=n)

	# si, sj, sk = AccValid_s.shape[0], AccValid_s.shape[1], AccValid_s.shape[2]


	dfBestAccValid_s=pd.DataFrame(BestAccValid_s)
	dfBestAccValid_s.plot.box(ax=axes[0,1])
	
	dfBestAceValid_s=pd.DataFrame(BestAceValid_s)
	dfBestAceValid_s.plot.box(ax=axes[1,1])
	
	dfBestMseValid_s=pd.DataFrame(BestMseValid_s)
	dfBestMseValid_s.plot.box(ax=axes[2,1])
	
	dfBestAccValid_s_wu=pd.DataFrame(BestAccValid_s_wu)
	dfBestAccValid_s_wu.plot.box(ax=axes[3,1])
	
	dfBestAceValid_s_wu=pd.DataFrame(BestAceValid_s_wu)
	dfBestAceValid_s_wu.plot.box(ax=axes[4,1])
	
	dfBestMseValid_s_wu=pd.DataFrame(BestMseValid_s_wu)
	dfBestMseValid_s_wu.plot.box(ax=axes[5,1])
	
	plt.show()
	
# 	ax = sns.heatmap(BestAccValid_s)
# 	ax1 = sns.heatmap(BestAccValid_s_wu)
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
	TAccValid_s = getattr(fnp.f, 'TensorAccValid_s')
	TAceValid_s = getattr(fnp.f, 'TensorAceValid_s')
	TMseValid_s = getattr(fnp.f, 'TensorMseValid_s')
	TAccValid_s_wu =  getattr(fnp2.f, 'TensorAccValid_s')
	TAceValid_s_wu =  getattr(fnp2.f, 'TensorAceValid_s')
	TMseValid_s_wu =  getattr(fnp2.f, 'TensorMseValid_s')

	sns.heatmap(BestAccValid_s, ax=axes[0])
	sns.heatmap(BestAccValid_s_wu, ax=axes[1])
	plt.show()

	# print(ggplot(mtcars, aes('mpg', 'qsec')) + geom_point(colour='steelblue') + scale_x_continuous(breaks=[10,20,30], labels=["horrible", "ok", "awesome"]))
	# plt.show()
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
	


