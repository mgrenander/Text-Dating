from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import time

start = time.time()

print('Loading Data...')
x = np.loadtxt("train_x.csv", delimiter=",")  # load from text
print('Loading Labels...')
y = np.loadtxt("train_y.csv", delimiter=",")
print('Loading Test Set...')
test = np.loadtxt("test_x.csv", delimiter=",")


threshhold = 230
x[x<threshhold] = 0
x[x>= threshhold] = 255

x = x.reshape(-1, 64, 64,1)  # reshape
# test[test<threshhold] = 0
# test[test>=threshhold] = 255

test = test.reshape(-1, 64, 64,1)

cls = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,24,25,27,28,30,32,35,36,40,42,45,48,49,54,56,63,64,72,81]


#makes the y values correspond to 1-40 (decode later)
newY =[]
for val in y:
    for idx,num in enumerate(cls):
        if val == num:
            newY.append(idx)


newY = np.asarray(newY)
y = newY.reshape(-1, 1)



train = x[:45000]
trainLabs = y[:45000]

val = x[45000:50000]
valLabs = y[45000:50000]

# basic idea:                               https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721
# old tutorial (not so relevant here) :     https://www.tensorflow.org/tutorials/layers
# new tutorial (where this code is from):   https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html


batch_size = 32     # in each iteration, we consider 32 training examples at once
num_epochs = 200    # we iterate 200 times over the entire training set
kernel_size = 3     # we will use 3x3 kernels throughout
pool_size = 2       # we will use 2x2 pooling throughout
conv_depth_1 = 32   # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64   # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5   # dropout in the FC layer with probability 0.5
hidden_size = 512   # the FC layer will have 512 neurons

#(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

X_train = train
y_train = trainLabs
X_test = val
y_test = valLabs

print(np.shape(X_train))
print(np.shape(y_train))

num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10 
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
