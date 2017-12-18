'''
-NEED TO ADJUST NUMBER OF NODES/LAYER based on data input request
-NEED TO select appropriate number of classes

'''
# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
# from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
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


# ---------------------- CNN parameters ----------------------
vocabulary= ['I','went','to','school','yesterday','wanted','talk','you']

num_classes=2
len_vocabulary=len(vocabulary)
words=5
convolution_stride=2
region_size=3
convoluted_window_height=int(((words+(2*(region_size-1)))-region_size)/(convolution_stride))+1

pooling_size=4/2
#region_size_2=5
num_weights=1000
pooling_units=100

#----------------------------- reading in data ------------------------

sc = SampleCreator(400)

# Category 7
samples7 = sc.get_samples(6)
label7 = sc.get_label(6)

print samples7
print label7
sys.exit()




# -------------------- fake testing data --------------------------------
#X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
# X=['I went to school yesterday'.split(), 'I wanted talk to you'.split(),'to talk you school I'.split(),'school to wanted you I'.split()] # 2D array
# X_train= sc.correct_input(vocabulary, X)
# y_train=[0,1,1,0] # y has to be a list of numbers
# y_train = to_categorical(y_train, num_classes) # One-hot encode the labels


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
	print (np.array(padded_matrix))
	
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
convoluted_input1=bag_of_words_conversion(X_train,region_size,convolution_stride,words,len_vocabulary)



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
'''
# ---------- all together in one go ... it works ---------------
model.fit(x=convoluted_input1, y=y_train)
'''
# ----------------- try to pass by batch -----------------------
# seq = ToDenseSeq(convoluted_input1[:2],y_train[:2],1)
# model.fit_generator(seq)

# seq = ToDenseSeq(convoluted_input1,y_train,1) # dummy test using the training same as testing
# print ('------------ this is the testing result!!!------------')
# print (model.evaluate_generator(seq)) # returns [loss_function_value, accuracy]
# print ("Time spent: {}s".format(time.time() -start))