# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:18:21 2021

@author: aguboshimec
"""

#General lib. imports
import keras
import copy
import numpy as np
from numpy import argmax, array, arange
import matplotlib.pyplot as plt
from keras import backend as K
from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense,Input,concatenate, UpSampling1D, Dropout, UpSampling2D, MaxPooling1D, Activation, GaussianNoise,BatchNormalization,  Flatten, ZeroPadding2D, Conv1D, Conv2D, MaxPooling2D, Lambda, Layer
from keras.utils import plot_model, to_categorical
from tensorflow import keras


noOfSamples = 5000
data_allA = None
data_allB = None
#Generate training dataset
def dataset():
    global data_allA, data_allB
    data_allA = []
    data_allB = []
    for i in range (noOfSamples):
        dataA = np.random.randint(low=0, high=2, size=(2, ))
        dataB = np.random.randint(low=2, high=4, size=(2, ))
        # the convert to one-hot encoded version
        data_allB.append(dataB)
        data_allA.append(dataA)        

dataset() 

data_allB = array(data_allB)
data_allA = array(data_allA)

#data set formatting. equally Splits dataset into training and validation:
data_allA = np.array_split(data_allA, 2)
x__trainA = data_allA[1] #training
x__validtnA = data_allA[0] #validation

x__trainA = np.reshape(data_allA[1], (len(x__trainA),2)) #training
x__validtnA = np.reshape(data_allA[0], (len(x__validtnA),2)) #validation

#data set formatting. equally Splits dataset into training and validation:
data_allB = np.array_split(data_allB, 2)
x__trainB = data_allB[1] #training
x__validtnB = data_allB[0] #validation

x__trainB = np.reshape(data_allB[1], (len(x__trainB),2)) #training
x__validtnB = np.reshape(data_allB[0], (len(x__validtnB),2)) #validation


labels = np.concatenate((x__trainA, x__trainB), axis=1)
val_labels = np.concatenate((x__validtnA, x__validtnB), axis=1)

left_branch_input = Input(shape=(2,), name='Left_input')
left_branch_output1 = Dense(6, activation='relu')(left_branch_input)

left_branch_output2 = Dense(2, activation='sigmoid')(left_branch_output1)

right_branch_input = Input(shape=(2,), name='Right_input')
right_branch_output1 = Dense(6, activation='relu')(right_branch_input)

right_branch_output2 = Dense(2, activation='sigmoid')(right_branch_output1)

concat = concatenate([left_branch_output2, right_branch_output2], name='Concatenate')
final_model_output = Dense(4, activation='linear')(concat)
final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                    name='Final_output')

final_model.compile(optimizer='adam', loss='mse')

final_model.summary()
plot_model(final_model, "modelconcat.png", show_shapes=True)
# To train
final_model.fit([x__trainA, x__trainB], labels, validation_data = ([x__validtnA, x__validtnB], val_labels),  epochs=50, batch_size= 50)

Tdata_allA = None 
Tdata_allB = None

def test_dataset():
    global Tdata_allA, Tdata_allB
    
    Tdata_allA = []
    Tdata_allB = []
    for i in range (noOfSamples):
        TdataA = np.random.randint(low=0, high=2, size=(2, ))
        TdataB = np.random.randint(low=2, high=4, size=(2, ))
        # the convert to one-hot encoded version
        Tdata_allA.append(TdataB)
        Tdata_allB.append(TdataA)        

test_dataset()

Tdata_allB = array(Tdata_allB)
Tdata_allA = array(Tdata_allA)


testVals = np.concatenate((Tdata_allB, Tdata_allA), axis=1)
decoded_dataA  = final_model.predict([Tdata_allB, Tdata_allA])

