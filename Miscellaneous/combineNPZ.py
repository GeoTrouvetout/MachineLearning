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

def main():
	parser = argparse.ArgumentParser(description="open [file.npz] and convert it into csv")
	parser.add_argument("FILESNPZ", help="fileName of the input array", nargs='+')
	parser.add_argument("-o", "--output-npz",dest="OUTNPZ", help="fileName of the input array")
	args = parser.parse_args()
	fnp0 = np.load(args.FILESNPZ[0])
# 	fnp2 = np.load(args.FILESNPZ)
	print(len(args.FILESNPZ))
	c={}
	for i in fnp0.keys():
		ft=[]
		for n in range(len(args.FILESNPZ)):
			fnp = np.load(args.FILESNPZ[n])
			k = getattr(fnp.f, i)
# 			k2 = getattr(fnp2, i)
			d = pd.DataFrame(k)
# 			df2 = pd.DataFrame(k2)
			ft.append(d)
			print(ft)
		df = pd.concat(ft, axis=1)
		c[i] = df
		
	np.savez(arg.OUTNPZ, **c)



if __name__ == "__main__":
	print("open and display [dlrecital] experiments results")
	main()
	





