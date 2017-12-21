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
from sklearn.metrics import confusion_matrix


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

def train_model_seara(train_file='pickle_all.pickle', **args):
	start = time.time()
	# Here put all the main training code in this function
	# file_stream = file_io.FileIO(train_file, mode='r')
	# X, y= pickle.load(file_stream)



	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	words=5 # number of words per sample
	num_classes=2
	region_size= 3
	convolution_stride=2
	num_weights=10
	pooling_units=20
	pooling_size=int((int(((words+(2*(region_size-1)))-region_size)/(convolution_stride))+1 )/pooling_units) +1
	len_vocabulary=5

	nb_epoch=8

	def correct_input_local(vocab, sentences):
		dic={}
		for index,item in enumerate(vocab):
			dic[item]=index
		result_sentences=[]
		for sent in sentences:
			sentence_matrix=[]
			for word in sent:
				temp=[0]*len(vocab)
				try: temp[dic[word]]=1
				except:
					pass
				sentence_matrix.append(temp)
			#matrix= np.array(sentence_matrix)
			matrix=sentence_matrix
			result_sentences.append(matrix)
		result_sentences=np.array(result_sentences)
		return result_sentences


	vocabulary= ['I','went','to','school','yesterday','wanted','talk','you'] # this is not actually needed, for fakes only
	len_vocabulary= len(vocabulary)
	X=['I went to school yesterday'.split(),'I love you like yesterday'.split(), 'I wanted talk to you'.split(),'to talk you school I'.split(),'school to wanted you I'.split()] # 2D array
	X= correct_input_local(vocabulary, X)
	num_classes=2
	y=[0,1,1,1,0] # y has to be a list of numbers
	y_temp= to_categorical(y, 2) # One-hot encode the labels
	X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)




	# ---------------------- bag of words concatenation -----------------------
	def padding(data,region_size,vocabulary_length):
		#result_sentences = np.ndarray(shape=(len(data)+2*(region_size-1),vocabulary),dtype=float)
		result_sentences=np.zeros((region_size-1, vocabulary_length))
		result_sentences=np.concatenate((result_sentences, data), axis=0)
		result_sentences=np.concatenate((result_sentences, np.zeros((region_size-1, vocabulary_length))), axis=0)

		# --------------------- convert to list of np arrays -----------------------------

		# gives an 2d numpy arrays
		return result_sentences

	def bag_of_words_convolution_persample(data,region_size,stride,num_words,len_vocabulary):
		padded_matrix=padding(data,region_size,len_vocabulary) # still a list of lists 2D
		window_height=int(math.ceil( float(padded_matrix.shape[0]-(region_size-1))/stride ))
		# print ("numpy window height: "+ str(window_height))
		result_matrix=np.zeros(shape=(window_height,padded_matrix.shape[1]))
		i=0
		j=0
		while i+(region_size-1) < padded_matrix.shape[0]:
			try:temp=np.sum(padded_matrix[i:i+region_size],axis=0)
			except:
				print ('--------------------')
				for j in range(i,i+region_size):
					print (padded_matrix[j])
				sys.exit('error!!')

			result_matrix[j]=(temp)
			i+=stride
			j+=1

		return result_matrix# original
			

	def bag_of_words_conversion(X_train,region_size,convolution_stride,words_in_sentence,len_vocabulary,stride):
		# assuming that X_train is numpy of numpy of numpy 
		padded_matrix_size = words_in_sentence+2*(region_size-1)
		window_height=int(math.ceil( float(padded_matrix_size-(region_size-1))/stride ))

		result=np.zeros(shape=(X_train.shape[0],window_height,len_vocabulary))

		i=0
		for sample in X_train:
			result[i]=(bag_of_words_convolution_persample(sample,region_size,convolution_stride,words_in_sentence,len_vocabulary))
			i+=1
		result=np.array(result)
		
		#print (result)
		return result

	#---------------------- CNN single -----------------------
	# convoluted_input_test=bag_of_words_conversion(X_test,region_size,convolution_stride,words,len_vocabulary,convolution_stride)
	# convoluted_input_train=bag_of_words_conversion(X_train,region_size,convolution_stride,words,len_vocabulary,convolution_stride)
	convoluted_window_height=X_train[0].shape[0] 

	# print (convoluted_input1[:5])
	# print (y_train[:5])
	# sys.exit()

	padded_matrix_size = words+2*(region_size-1)
	# convoluted_window_height=int(math.ceil( float(padded_matrix_size-(region_size-1))/convolution_stride ))



	input_1 = Input(shape=(convoluted_window_height,len_vocabulary)) # height, width, depth
	#input_2 = Input(shape=(None,len_vocabulary)) # height, width, depth
	conv1d_1 = Conv1D(num_weights,1,activation='relu',padding='same')(input_1)
	#conv1d_2 = Conv1D(num_weights,1,activation='relu',padding='same')(input_1)
	max_pooling1d_1 = AveragePooling1D(pool_size=pooling_size)(conv1d_1)

	# conv1d_2 = Conv1D(num_weights,1,activation='relu')(max_pooling1d_1)
	# max_pooling1d_2 = AveragePooling1D(pool_size=pooling_size)(conv1d_2)


	#flatten_1 = Flatten()(max_pooling1d_2)
	flatten_1 = Flatten()(max_pooling1d_1)
	dense_1 = Dense(words,activation='relu')(flatten_1)
	dense_2 = Dense(num_classes, activation="softmax")(dense_1)

	model = Model(inputs=input_1, outputs=dense_2)

	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
	              optimizer='adam', # using the Adam optimiser
	              metrics=['accuracy']) # reporting the accuracy

	print(model.summary())

	# ---------- all together in one go ... it works ---------------
	# model.fit(x=convoluted_input1, y=y_train) original
	# seq = ToDenseSeq(convoluted_input_train,y_train, 4)
	# model.fit_generator(seq)
	# seq = ToDenseSeq(convoluted_input_test,y_test,1) # dummy test using the training same as testing
	# print ('------------ this is the testing result!!!------------')
	# print (model.evaluate_generator(seq)) # returns [loss_function_value, accuracy]
	# print ("Time spent: {}s".format(time.time() -start))


	# model.fit_generator(generator=batch_generator(X_train, y_train, 25),
	#                     # nb_epoch=nb_epoch, 
	#                     samples_per_epoch=X_train.shape[0])
	# print (model.evaluate_generator(generator=batch_generator(X_test, y_test, 25),steps=100))
	model.fit(X_train,y_train)
	a=model.predict(X_train)
	print a
	a=np.array(a.argmax(axis=1))
	
	# then store the a 

	y=[0,1,1,1]
	print a
	print y_train

	conf_matrix=confusion_matrix(y, a)

	print conf_matrix

if __name__ == '__main__':
	# train_model_seara()
	a=np.array([1,2,3,4])
	# np.savetxt('test.out',a)
	b= np.loadtxt('test.out')
	print a
	print b

