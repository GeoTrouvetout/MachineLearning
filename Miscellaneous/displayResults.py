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
import matplotlib.pyplot  as pyplot
from ggplot import *

plt.style.use('ggplot')
########### FUNCTIONS DEF ################


def read_npz(filename):
	file = np.load(filename)
	return file


def main():
	parser = argparse.ArgumentParser(description="open [file.npz] and convert it into csv")
	parser.add_argument("FILENPZ1", help="fileName of the input array")
	parser.add_argument("FILENPZ2", help="fileName of the input array 2")
	parser.add_argument("FILENPZ3", help="fileName of the input array 3")
	parser.add_argument("FILENPZ4", help="fileName of the input array 4")

	args = parser.parse_args()

	fnp1 = read_npz(args.FILENPZ1)
	fnp2 = read_npz(args.FILENPZ2)
	fnp3 = read_npz(args.FILENPZ3)
	fnp4 = read_npz(args.FILENPZ4)

	# pd.DataFrame(data=fnp.f[1:,1:],index=data[1:,0],columns=data[0,1:])
	# print(fnp.keys())

	header = fnp1.keys()
	nameHeader = ["OptNbSample_ns", "OptMseTrain_ns", "OptMseValid_ns","TensorMseTrain_ns","TensorMseValid_ns","OptNbSample_s","OptAccTrain_s","OptAceTrain_s","OptMseValid_s","OptAccValid_s","OptAceValid_s","TensorMseValid_s","TensorAceTrain_s","TensorAceValid_s","TensorAccValid_s","ArrayAccTest","ArrayAceTest","ArrayMseTest"]
	nbMatrix = 0
	nbTensor = 0
	
	ylim = (0.05, 1.05)

	AccTest1 = getattr(fnp1.f, 'ArrayAccTest').T
	AccTest2 = getattr(fnp2.f, 'ArrayAccTest').T
	AccTest3 = getattr(fnp3.f, 'ArrayAccTest').T
	AccTest4 = getattr(fnp4.f, 'ArrayAccTest').T

	dfAccTest1 = pd.DataFrame(AccTest1, columns=range(100,-10,-10))
	dfAccTest2 = pd.DataFrame(AccTest2, columns=range(9,0,-1))

	dfAccTest3 = pd.DataFrame(AccTest3, columns=range(100,-10,-10))
	dfAccTest4 = pd.DataFrame(AccTest4, columns=range(9,0,-1))

	fT = [dfAccTest1,dfAccTest2]
	dfAccTest = pd.concat(fT, axis=1)
	fTwu = [dfAccTest3,dfAccTest4]
	dfAccTest_wu = pd.concat(fTwu, axis=1)

	# print(dfAccTest3)
	# print(dfAccTest4)

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10), sharey=True)

	dfAccTest= dfAccTest.reindex_axis(sorted(dfAccTest.columns,reverse=True), axis=1)
	dfAccTest.plot.box(ax=axes[0], title='Accuracy TEST with CAE')

	# dfAccTest2.plot.box(ax=axes[0,1], ylim=ylim)
	dfAccTest_wu = dfAccTest_wu.reindex_axis(sorted(dfAccTest_wu.columns, reverse=True), axis=1)
	dfAccTest_wu.plot.box(ax=axes[0],title='Accuracy TEST with CNN')
	# dfAccTest4.plot.box(ax=axes[1,1], ylim=ylim)
	print(dfAccTest)
	print(dfAccTest_wu)

	dfDiff = dfAccTest - dfAccTest_wu
	# dfAccTest_wu = dfAccTest_wu.reindex_axis(sorted(dfAccTest_wu.columns, reverse=True), axis=1)
	# ylimd = (-0.1, 0.1)
	dfDiff.plot.box(ax=axes[1],title='Accuracy TEST with CNN')
	plt.show()
# 	ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
	# xlim = (-1, 11)

	# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(6, 6), sharey=True)
	# axs = axs.ravel()

	# i=0
	# for i, h in enumerate(['OptAccValid_s', 'ArrayAccTest']):
	# 	# arr_i = '%s%s' % ('arr_', i)
	# 	k1 = getattr(fnp1.f, h)
	# 	df1=pd.DataFrame(k1.T)
	# 	print(df1	)
	# 	k2 = getattr(fnp2.f, h)
	# 	df2=pd.DataFrame(k2.T)

	# 	print(h, k1.shape, k2.shape)
		
	# 	df1.plot(kind='box', ylim=ylim, ax=axs[i])
	# 	df2.plot(kind='box', ylim=ylim, ax=axs[i])


# 			axes[]plot.box(ax=axes[0,0])

		# if len(k.shape) == 3:
			# nbTensor += 1
			# km = k.reshape(k.shape[0]*k.shape[1], k.shape[2])
			# for i in np.arange(k.shape[0]) 
			# 	k[i,:,:]
			# print(km.shape)

	# plt.show()
	# print(nbMatrix)
	# print(nbTensor)

# 	axs = axs.ravel()

# 	xlim = (-10, 10)
# 	ylim = (-1, 2)
# 	alpha = 0.3




# 	AccTest = getattr(fnp.f, 'ArrayAccTest').T
# 	AccTest_wu = getattr(fnp2.f, 'ArrayAccTest').T
# 	# AccTest_wu_l = getattr(fnp3.f, 'ArrayAccTest').T
# 	AceTest = getattr(fnp.f, 'ArrayAceTest').T
# 	AceTest_wu = getattr(fnp2.f, 'ArrayAceTest').T
# 	MseTest = getattr(fnp.f, 'ArrayMseTest').T
# 	MseTest_wu = getattr(fnp2.f, 'ArrayMseTest').T
# 	# MseTest_wu_l = getattr(fnp3.f, 'ArrayMseTest').T

# 	# fig, axes = plt.subplots(nrows=3, figsize=(6, 6), sharey=True)
# 	AccValid_s = getattr(fnp.f, 'OptAccValid_s').T
# 	AccValid_s_wu = getattr(fnp2.f, 'OptAccValid_s').T
# 	# AccValid_s_wu_l = getattr(fnp3.f, 'OptAccValid_s').T

# 	AceValid_s = getattr(fnp.f, 'OptAceValid_s').T
# 	AceValid_s_wu = getattr(fnp2.f, 'OptAceValid_s').T
# 	MseValid_s = getattr(fnp.f, 'OptMseValid_s').T
# 	MseValid_s_wu = getattr(fnp2.f, 'OptMseValid_s').T
# # 	# for i in range(OptAccValid_s.shape[1]):
# # 	# 

# # # 	plt.ylim(-.05, 1.05)
# # # 	seqsig = np.arange(0,110,10)
# 	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharey=True)
# 	ylim = (0.0, 1.05)
# 	alpha = 0.3
	
# 	dfAccValid_s=pd.DataFrame(AccValid_s, columns=range(100,-10,-10))
	
# 	dfAccValid_s.plot.box(ax=axes[0,0], ylim=ylim)

# 	dfAccValid_s_wu=pd.DataFrame(AccValid_s_wu, columns=range(100,-10,-10))
# 	# dfAccValid_s_wu_l=pd.DataFrame(AccValid_s_wu_l, columns=range(9,0,-1))
# 	frames = [dfAccValid_s_wu,dfAccValid_s_wu_l] 
# 	dfAccValid_s_wuT = pd.concat(frames, axis=1)
# 	dfAccValid_s_wuT.plot.box(ax=axes[0,1], ylim=ylim)
# 	print(dfAccValid_s_wu)
# 	# print(dfAccValid_s_wu_l)
# 	print(dfAccValid_s_wuT)

# 	dfAccTest=pd.DataFrame(AccTest, columns=range(100,-10,-10))
# 	dfAccTest.plot.box(ax=axes[1,0])

# 	dfAccTest_wu=pd.DataFrame(AccTest_wu, columns=range(100,-10,-10))
# 	# dfAccTest_wu_l=pd.DataFrame(AccTest_wu_l, columns=range(9,0,-1))

# 	frames = [dfAccTest_wu,dfAccTest_wu_l]
# 	dfAccTest_wuT = pd.concat(frames)
# 	dfAccTest_wuT.plot.box(ax=axes[1,1], ylim=ylim)

# 	plt.show()



	# fig1, axes1 = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
	# ylim = (0.90, 1.05)
	# alpha = 0.3
	# # fig1.set_yscale('log')
	# # seqs = range(100, -10, 10)
	# # meanAccValid_s = dfAccValid_s.mean(axis=0)  #.plot(ax=axes[0,1])	
	# # print(meanAccValid_s.shape)
	# # meanAccValid_s.plot(kind='line',, ax=axs[0,0], xlim=xlim, ylim=ylim)
	# dfAccValid_s=pd.DataFrame(AccValid_s)
	# dfAccValid_s.plot.box(ax=axes1[0,0], ylim=ylim)

	# dfAceValid_s=pd.DataFrame(AceValid_s)
	# dfAceValid_s.plot.box(ax=axes1[0,1])
	
	# dfMseValid_s=pd.DataFrame(MseValid_s)
	# dfMseValid_s.plot.box(ax=axes1[0,2])

	# dfAccTest=pd.DataFrame(AccTest)
	# dfAccTest.plot.box(ax=axes1[1,0])

	# dfAceTest=pd.DataFrame(AceTest)
	# dfAceTest.plot.box(ax=axes1[1,1])

	# dfMseTest=pd.DataFrame(MseTest)
	# dfMseTest.plot.box(ax=axes1[1,2])


	# fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
	# # fig2.set_yscale('log')

	# dfAccValid_s_wu=pd.DataFrame(AccValid_s_wu)
	# dfAccValid_s_wu.plot.box(ax=axes2[0,0], ylim=ylim)
	# dfAceValid_s_wu=pd.DataFrame(AceValid_s_wu)
	# dfAceValid_s_wu.plot.box(ax=axes2[0,1])

	# dfMseValid_s_wu=pd.DataFrame(MseValid_s_wu)
	# dfMseValid_s_wu.plot.box(ax=axes2[0,2])

	# dfAccTest_wu=pd.DataFrame(AccTest_wu)
	# dfAccTest_wu.plot.box(ax=axes2[1,0], ylim=ylim)

	# dfAceTest_wu=pd.DataFrame(AceTest_wu)
	# dfAceTest_wu.plot.box(ax=axes2[1,1])

	# dfMseTest_wu=pd.DataFrame(MseTest_wu)
	# dfMseTest_wu.plot.box(ax=axes2[1,2])



	# plt.show()

	# n=1
	# BestAccValid_s = np.mean( getattr(fnp.f, 'TensorAccValid_s') , axis=n)
	# BestAceValid_s = np.mean( getattr(fnp.f, 'TensorAceValid_s') , axis=n)
	# BestMseValid_s = np.mean( getattr(fnp.f, 'TensorMseValid_s') , axis=n)
	
	# BestAccValid_s_wu = np.mean( getattr(fnp2.f, 'TensorAccValid_s') , axis=n)
	# BestAceValid_s_wu = np.mean( getattr(fnp2.f, 'TensorAceValid_s') , axis=n)
	# BestMseValid_s_wu = np.mean( getattr(fnp2.f, 'TensorMseValid_s') , axis=n)

	# si, sj, sk = AccValid_s.shape[0], AccValid_s.shape[1], AccValid_s.shape[2]


# 	dfBestAccValid_s=pd.DataFrame(BestAccValid_s)
# 	dfBestAccValid_s.plot.box(ax=axes[0,0])
# 	
# 	dfBestAceValid_s=pd.DataFrame(BestAceValid_s)
# 	dfBestAceValid_s.plot.box(ax=axes[0,1])
# 	
# 	dfBestMseValid_s=pd.DataFrame(BestMseValid_s)
# 	dfBestMseValid_s.plot.box(ax=axes[0,2])
# 	
# 	dfBestAccValid_s_wu=pd.DataFrame(BestAccValid_s_wu)
# 	dfBestAccValid_s_wu.plot.box(ax=axes[0,3])
# 	
# 	dfBestAceValid_s_wu=pd.DataFrame(BestAceValid_s_wu)
# 	dfBestAceValid_s_wu.plot.box(ax=axes[0,4])
# 	
# 	dfBestMseValid_s_wu=pd.DataFrame(BestMseValid_s_wu)
# 	dfBestMseValid_s_wu.plot.box(ax=axes[0,5])
# 	
	
# 	ax = sns.heatmap(BestAccValid_s)
# 	ax1 = sns.heatmap(BestAccValid_s_wu)
# 	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
	# TAccValid_s = getattr(fnp.f, 'TensorAccValid_s')
	# TAceValid_s = getattr(fnp.f, 'TensorAceValid_s')
	# TMseValid_s = getattr(fnp.f, 'TensorMseValid_s')
	# TAccValid_s_wu =  getattr(fnp2.f, 'TensorAccValid_s')
	# TAceValid_s_wu =  getattr(fnp2.f, 'TensorAceValid_s')
	# TMseValid_s_wu =  getattr(fnp2.f, 'TensorMseValid_s')

	# sns.heatmap(BestAccValid_s, ax=axes[0])
	# sns.heatmap(BestAccValid_s_wu, ax=axes[1])
	# plt.show()

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
	


