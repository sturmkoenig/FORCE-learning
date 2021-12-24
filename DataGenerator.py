#!/usr/bin/env python

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import Sequence
from column import get_default_pconn, run_column_model
from scipy.fft import irfft


class DataGenerator(Sequence):

	def __init__(self,list_IDs, batch_size=256,dim=(40,600),dim_param=(1600), n_channels=1, n_classes=10, shuffle=False):
		self.dim = dim
		self.dim_param = dim_param
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.index = 0
		self.on_epoch_end()

	def on_epoch_end(self):
		self.index=0
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, current_ID):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = (irfft(np.load(os.path.join('/scratch2/llauer/tf_dat',current_ID))['data'],n=600,axis=1)).T
		Y = np.load(os.path.join('/scratch2/llauer/tf_dat',current_ID))['pconn']
		Y = np.reshape(Y, (256,1600))
		return X, Y

	def __len__(self):
		'denotes the number of batches per epoch'
		return len(self.list_IDs)-1

	def __getitem__(self,index):
		'Generate one batch of data'

		# Find list of IDs
		current_ID = self.list_IDs[self.indexes[self.index]]
		self.index = self.index+1

		# Generate data
		X, Y = self.__data_generation(current_ID)

		return X,Y
