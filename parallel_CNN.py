'''
-NEED TO ADJUST NUMBER OF NODES/LAYER based on data input request
-NEED TO select appropriate number of classes

'''
# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
# from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
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
from keras.backend import temporal_padding
import math
import time
import sys
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

start = time.time()

def correct_input(vocab, sentences):
	dic={}
	for index,item in enumerate(vocab):
		dic[item]=index
	result_sentences=[]
	for sent in sentences:
		sentence_matrix=[]
		for word in sent:
			temp=[0]*len(vocab)
			temp[dic[word]]=1
			sentence_matrix.append(temp)
		#matrix= np.array(sentence_matrix)
		matrix=sentence_matrix
		result_sentences.append(matrix)
	#result_sentences=np.array(result_sentences)
	return result_sentences

# ---------------------- CNN parameters ----------------------
vocabulary= ['I','went','to','school','yesterday','wanted','talk','you']

num_classes=2
len_vocabulary=len(vocabulary)
words=5

convolution_stride_1=2
region_size_1=3
convoluted_window_height_1=int(((words+(2*(region_size_1-1)))-region_size_1)/(convolution_stride_1))+1

convolution_stride_2=2
region_size_2=4
convoluted_window_height_2=int(((words+(2*(region_size_2-1)))-region_size_2)/(convolution_stride_2))+1

pooling_units=2
pooling_size_1=convoluted_window_height_1/pooling_units

pooling_size_2=convoluted_window_height_2/pooling_units
#region_size_2=5

num_weights=1


#----------------------------- reading in data ------------------------

#X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
X=['I went to school yesterday'.split(), 'I wanted talk to you'.split(),'to talk you school I'.split(),'school to wanted you I'.split()] # 2D array
X_train= correct_input(vocabulary, X)
y_train=[0,1,1,1] # y has to be a list of numbers
y_train = to_categorical(y_train, num_classes) # One-hot encode the labels


# ---------------------- bag of words concatenation -----------------------
def padding(data,region_size,vocabulary_length):
	#result_sentences = np.ndarray(shape=(len(data)+2*(region_size-1),vocabulary),dtype=float)	
	result_sentences=[[0]*vocabulary_length for i in range(region_size-1)]
	result_sentences.extend(data)
	result_sentences.extend([[0]*vocabulary_length for i in range(region_size-1)])

	# --------------------- convert to list of np arrays -----------------------------
	final_result=[]
	for line in result_sentences:
		temp=np.array(line)
		final_result.append(temp)

	return final_result

def bag_of_words_convolution_persample(data,region_size,stride,num_words,len_vocabulary):
	'''
	accept: data list of list form 
	return np.ndarray for an image
	'''
	padded_matrix=padding(data,region_size,len_vocabulary) # still a list of lists 2D
	print np.array(padded_matrix)
	
	result_matrix=[]
	i=0
	while i+(region_size-1) < len(padded_matrix):
		temp=np.sum(padded_matrix[i:i+region_size],axis=0)
		result_matrix.append(temp)
		i+=stride
	result_matrix=np.array(result_matrix)
	return np.array(result_matrix)

def bag_of_words_conversion(X_train,region_size,convolution_stride,words_in_sentence,len_vocabulary):
	result=[]
	for sample in X_train:
		result.append(bag_of_words_convolution_persample(sample,region_size,convolution_stride,words_in_sentence,len_vocabulary))

	result=np.array(result)
	return result

# ---------------------- CNN single -----------------------
# options: more layers
# more parallels
convoluted_input1=bag_of_words_conversion(X_train,region_size_1,convolution_stride_1,words,len_vocabulary)
convoluted_input2=bag_of_words_conversion(X_train,region_size_2,convolution_stride_2,words,len_vocabulary)



input_1 = Input(shape=(convoluted_window_height_1,len_vocabulary)) # height, width, depth
input_2 = Input(shape=(convoluted_window_height_2,len_vocabulary)) # height, width, depth
conv1d_1 = Conv1D(num_weights,1,activation='relu',padding='same')(input_1)
conv1d_2 = Conv1D(num_weights,1,activation='relu',padding='same')(input_2)
#conv1d_2 = Conv1D(num_weights,1,activation='relu',padding='same')(input_1)
max_pooling1d_1 = AveragePooling1D(pool_size=pooling_size_1)(conv1d_1)
max_pooling1d_2 = AveragePooling1D(pool_size=pooling_size_2)(conv1d_2)

merge_1 = Concatenate(axis=1)([max_pooling1d_1,max_pooling1d_2])


# conv1d_2 = Conv1D(num_weights,1,activation='relu')(max_pooling1d_1)
# max_pooling1d_2 = AveragePooling1D(pool_size=pooling_size)(conv1d_2)

flatten_1 = Flatten()(merge_1)
dense_1 = Dense(words,activation='relu')(flatten_1)
dense_2 = Dense(num_classes, activation="softmax")(dense_1)

model = Model(inputs=[input_1,input_2], outputs=dense_2)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

print(model.summary())

model.fit(x=[convoluted_input1,convoluted_input1], y=y_train)

print ("Time spent: {}s".format(time.time() -start))