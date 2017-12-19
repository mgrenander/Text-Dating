'''
-NEED TO ADJUST NUMBER OF NODES/LAYER based on data input request
-NEED TO select appropriate number of classes

'''
# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
# from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import data_preprocessor
from sample_creator import SampleCreator
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

start = time.time()


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

# ------------------------ Matt new ---------------------------
pickle_all = open("Pickles/pickle_all.pickle", 'rb')
X_train, y_train = pickle.load(pickle_all)
print(len(X_train))
print(len(y_train))


# Small test sample
pickle_in = open("Pickles/test_pickle.pickle", 'rb')
X_train, y_train = pickle.load(pickle_in)
print (len(X_train))
print (len(y_train))
sys.exit()
print ('------- on to my part -----------')
# print(X_train)

# Verify types
# print(type(X_train))
# print(type(X_train[0]))
# print(type(X_train[0][0]))
#
# print(type(y_train))
# print(type(y_train[0]))

# All samples
# pickle_all = open("Pickles/samples_labels.pickle", 'rb')

# X_all, y_all = pickle.load(pickle_all)

# # Verify types
# print(type(X_all))
# print(type(X_all[0]))
# print(type(X_all[0][0]))

# print(type(y_all))
# print(type(y_all[0]))

# sys.exit()

#----------------------------- reading in data ------------------------
words=5 # number of words per sample
# sc = SampleCreator(words) # was 100

# # dummy Category 8
# samples8 = sc.get_samples(8)
# samples8=np.array(samples8)
# label8 = sc.get_label(8)
# print (type(samples8))
# print (samples8[0].shape)
# print (samples8[0])

# X_train=samples8[:5]
# y_train=label8
# print (len(X_train))
# print (y_train)

# ---------------------- CNN parameters ----------------------
def calculated_convoluted_window_height(region_size,convolution_stride):
	expanded=words+(2*(region_size-1))
	nums_windows=(expanded-(region_size-1))/convolution_stride
	return nums_windows


num_classes=9
convolution_stride=2
region_size=3
num_weights=1000
pooling_units=100
pooling_size=int((int(((words+(2*(region_size-1)))-region_size)/(convolution_stride))+1 )/pooling_units) +1

len_vocabulary=data_preprocessor.get_vocab_len()
#len_vocabulary=len(vocabulary)

# Category 7

# -------------------- fake testing data --------------------------------
# def correct_input_local(vocabulary, sentences):
# 	dic = {}
# 	for index, item in enumerate(vocabulary):
# 		dic[item] = index
# 	result_sentences=[]
# 	for sent in sentences:
# 		sentence_matrix = []
# 		for word in sent:
# 			temp = [0] * len(vocabulary)
# 			try:
# 				temp[dic[word]] = 1
# 			except KeyError:
# 				pass
# 			sentence_matrix.append(temp)
# 		matrix= sentence_matrix
# 		result_sentences.append(matrix)
       
# 	#result_sentences=np.array(result_sentences)
# 	return result_sentences
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
	#result_sentences=np.array(result_sentences)
	return result_sentences


# X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
vocabulary= ['I','went','to','school','yesterday','wanted','talk','you'] # this is not actually needed, for fakes only
len_vocabulary= len(vocabulary)
X=['I went to school yesterday'.split(),'I love you like yesterday'.split(), 'I wanted talk to you'.split(),'to talk you school I'.split(),'school to wanted you I'.split()] # 2D array
X_train= correct_input_local(vocabulary, X)
# print (type(X_train_seara))
# print (type(X_train_seara[0]))
# print (type(X_train_seara[0][0]))
# print (type(X_train_seara[0][0][0]))
num_classes=2
y_train=[0,1,1,1,0] # y has to be a list of numbers
y_train = to_categorical(y_train, 2) # One-hot encode the labels
# print (y_train)
print ('----generated so far by seara-----')


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
	#print (np.array(padded_matrix))
	
	result_matrix=[]
	i=0
	while i+(region_size-1) < len(padded_matrix):
		try:temp=np.sum(padded_matrix[i:i+region_size],axis=0)
		except:
			print ('--------------------')
			for j in range(i,i+region_size):
				print (padded_matrix[j])
			sys.exit('error!!')

		result_matrix.append(temp)
		i+=stride
	result_matrix=np.array(result_matrix) # original
	return np.array(result_matrix) # original
	

def bag_of_words_conversion(X_train,region_size,convolution_stride,words_in_sentence,len_vocabulary):
	result=[]
	for sample in X_train:
		result.append(bag_of_words_convolution_persample(sample,region_size,convolution_stride,words_in_sentence,len_vocabulary))

	result=np.array(result)
	print (result)
	return result

# ---------------------- CNN single -----------------------

# options: more layers
# more parallels
#len_vocabulary=data_preprocessor.get_vocab_len()
convoluted_input1=bag_of_words_conversion(X_train,region_size,convolution_stride,words,len_vocabulary)
# print (convoluted_input1.shape)
# sys.exit()


convoluted_window_height=convoluted_input1[0].shape[0] 

# print (convoluted_input1[:5])
# print (y_train[:5])
# sys.exit()



input_1 = Input(shape=(convoluted_window_height,len_vocabulary)) # height, width, depth
#input_2 = Input(shape=(None,len_vocabulary)) # height, width, depth
conv1d_1 = Conv1D(num_weights,1,activation='relu',padding='same')(input_1)
#conv1d_2 = Conv1D(num_weights,1,activation='relu',padding='same')(input_1)
max_pooling1d_1 = AveragePooling1D(pool_size=pooling_size)(conv1d_1)

conv1d_2 = Conv1D(num_weights,1,activation='relu')(max_pooling1d_1)
max_pooling1d_2 = AveragePooling1D(pool_size=pooling_size)(conv1d_2)


flatten_1 = Flatten()(max_pooling1d_2)
#flatten_1 = Flatten()(max_pooling1d_1)
dense_1 = Dense(words,activation='relu')(flatten_1)
dense_2 = Dense(num_classes, activation="softmax")(dense_1)

model = Model(inputs=input_1, outputs=dense_2)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

print(model.summary())

# ---------- all together in one go ... it works ---------------
model.fit(x=convoluted_input1, y=y_train)

# ----------------- try to pass by batch -----------------------
# seq = ToDenseSeq(convoluted_input1[:2],y_train[:2],1)
# model.fit_generator(seq)

# seq = ToDenseSeq(convoluted_input1,y_train,1) # dummy test using the training same as testing
# print ('------------ this is the testing result!!!------------')
# print (model.evaluate_generator(seq)) # returns [loss_function_value, accuracy]
# print ("Time spent: {}s".format(time.time() -start))
