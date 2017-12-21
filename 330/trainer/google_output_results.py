'''
-NEED TO ADJUST NUMBER OF NODES/LAYER based on data input request
-NEED TO select appropriate number of classes

'''
# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
# from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import argparse
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input
from keras.layers import AveragePooling1D
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Concatenate
from keras.models import Model
from keras.utils import Sequence
import math
import time
import sys
import pickle
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from sklearn.cross_validation import train_test_split



'''
trainsforming and put into dense one by one 
'''
# have to hard code- code the size of vocabulary!! 
# the pickle called final_pickle


class ToDenseSeq(Sequence):

	def __init__(self, x_set, y_set, batch_size):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):

		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		return batch_x,np.array(batch_y)

	def on_epoch_end(self):
	    pass

def model_write(train_file='pickle_all.pickle', **args):
	start = time.time()
	# Here put all the main training code in this function

	# f = file_io.FileIO(train_file, mode='w')
	# X, y= pickle.load(file_stream)
	directory="gs://text-dating/results/test.out"
	a=np.array([1,2,3,4])
	np.savetxt('test.out',a)


	return



	

if __name__ == '__main__':
	model_write()
	parser = argparse.ArgumentParser()
	# Input Arguments
	parser.add_argument(
		'--train-file',
		help='GCS or local paths to training data',
		required=True
	)

	parser.add_argument(
	  '--job-dir',
	  help='GCS location to write checkpoints and export models',
	  required=True
	)



	args = parser.parse_args()
	arguments = args.__dict__
	job_dir = arguments.pop('job_dir')

	# # train_model_seara(**arguments)

