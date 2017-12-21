'''
-NEED TO ADJUST NUMBER OF NODES/LAYER based on data input request
-NEED TO select appropriate number of classes

'''
# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
# from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import argparse
import math
import time
import sys
import pickle
#from tensorflow.python.lib.io import file_io
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix
from tensorflow.python.lib.io import file_io


# word count of text classification
# X:  an overall 2D sparse matrix
# y:  a 1D numpy array



def train_model_seara(train_file='pickle_.pickle', **args):
	start = time.time()
	# Here put all the main training code in this function
	file_stream = file_io.FileIO(train_file, mode='r')
	X, y= pickle.load(file_stream)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# ------------ testing input format ---------------------------
	def correct_input_local(vocab, sentences):
		dic={}
		for index,item in enumerate(vocab):
			dic[item]=index
		result_sentences=[]
		for sent in sentences:
			sentence_matrix=[0]*len(vocab)
			for word in sent:
				try:sentence_matrix[dic[word]]+=1
				except: pass
			result_sentences.append((sentence_matrix))
		result_sentences=csr_matrix(result_sentences)
		return result_sentences


	# vocabulary= ['I','went','to','school','yesterday','wanted','talk','you'] # this is not actually needed, for fakes only
	# # len_vocabulary= len(vocabulary)
	# X=['I went to school yesterday'.split(),'I love you like yesterday'.split(), 'I wanted talk to you'.split(),'to talk you school I'.split(),'school to wanted you I'.split()] # 2D array
	# X= correct_input_local(vocabulary, X)
	# # num_classes=9
	# y=np.array([0,1,1,1,0]) # y has to be a list of numbers
	# y= to_categorical(y_train, 9) # One-hot encode the labels


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	print X_train.shape

	clf = MultinomialNB()
	clf.fit(X_train, y_train)

def convert_to_numpy(train_file='nb.pickle', **args):
		# Here put all the main training code in this function

	file_stream = file_io.FileIO(train_file, mode='r')
	X, y= pickle.load(file_stream)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train=X_train.toarray()
	X_test=X_test.toarray()


	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	print clf.score(X_test,y_test)

if __name__ == '__main__':

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

	#train_model_seara(**arguments)
	convert_to_numpy(**argument)
