"""
Created on Sun Dec 2 11:28:29 2020
'So, I tried to evaluate the performance of several flavours of Autoencoders on Alamouti Coded Symbol (continous-valued data)'
'The following graphically results were performed on same dataset, epoch value and batchsize'
#Evaluation was based on: MSE vs. SNR & Loss vs. Epoch Value only.
@author: aguboshimec
""" 

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import keras
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split #split dataset into training and test data sets
from keras import layers, regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense,Input, Dropout, Activation, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Lambda, Layer, Add, Multiply
from keras.utils import plot_model
from pandas import DataFrame


#Variable Declaration and Assignment
tx_antenna = 2 #number of tx_antennas
rx_antenna = 2 #number of rx_antennas
dim = 2  #NB: dim has to be a value which is a factor of 'noOfSamples'.
noOfSamples = int(240000) #number of observations/sample data generated in the for-loop. It determines the 3 dimension of our matrix or vector.

if noOfSamples % 2==0:     #just a control measure to be sure ##noOfSamples is divisible be 2 always.
    noOfSamples = noOfSamples
else:
    noOfSamples = noOfSamples + 1

idx = int(noOfSamples/2) # the loop index will be the number of the splitted parts of the 3D tensor/array
bs = 20
epoch = 150
#to be able to reeuse the content of the variable, we set them as global variables: 
#Recall, the varibales are within a function, so I got to say they are global variables.
global Symbol_est_all_split, Symobl_all_split, Symbol_test_all, snr

snr = None
Symobl_all_split = None
Symbol_est_all_split = None
test_Symbol_est_all = None
DeepNN = None
pred_test_Symbol_est__allA = None
pred_test_Symbol_est__allB = None
pred_test_Symbol_est__allC = None
mean_sqaure_error_A = None
mean_sqaure_error_B = None
autoencoder_wo = None
autoencoder_w =None
variational_autoencoder = None
model = None

X_all = []
Symbol_all = [] #all transmitted symbol
Symbol_est_all = [] #all recieved or estimated symbols
Symbol_test_all = [] #all test symbols for evaluation of trained model
test_Noise = []  #all incremental noise for evaluation of testing model
test_Symbol_est_all = [] ##all noisy recieved or estimated symbols
mean_sqaure_error_A = []
mean_sqaure_error_B = []
mean_sqaure_error_C = []
mean_sqaure_error_D = []
pred_test_Symbol_est_all = []
pred_test_Symbol_est__allB = []
pred_test_Symbol_est__allC = []

#Generating dataset
for i in range(noOfSamples):
    
    #to generate the symbols (continous symbols (antenna 1 and 2 respectively) for now for testing purpose)
    x1 = np.around(np.random.randn(), 4) #random number drawn from normal distribution, rounded to 5 decimal places.
    x2 = np.around(np.random.randn(), 4) #same as above.
    Symbol = np.array([[x1], [x2]]) #for now, it is continous
    Symbol_all.append(Symbol)
    
    #Generate complex Channel Coefficients individually:
    h11 =  np.around(np.random.randn() - np.random.randn()*1j, 4)
    h12 =  np.around(np.random.randn() - np.random.randn()*1j, 4)
    h22 =  np.around(np.random.randn() - np.random.randn()*1j, 4)
    h21 =  np.around(np.random.randn() - np.random.randn()*1j, 4)
    Channel =  np.array([[h11, h12], [h21, h22],[h12.conj(), -(h11.conj())], [h22.conj(), -(h21.conj())]]) #the Channel Matrix
    #Generate AWGN Noise coefficients are each respective time slots (a & b):
    n1_a =  np.around(np.random.randn() - np.random.randn()*1j, 4)
    n2_a =  np.around(np.random.randn() - np.random.randn()*1j, 4)
    n1_b =  np.around(np.random.randn() - np.random.randn()*1j, 4)  
    n2_b =  np.around(np.random.randn() - np.random.randn()*1j, 4)  
    Noise = np.array([[n1_a], [n2_a], [n1_b.conj()], [n2_b.conj()]])

    
    y = np.add(np.dot(Channel,Symbol),Noise) #received signal for both time slots, both antennas
    Symbol_est = np.dot(np.linalg.pinv(Channel), y)
    Symbol_est_all.append(Symbol_est.real)

    #Obtains a testing dataset to evaluate the effectiveness of the trained model:
    #to generate the symbols (continous symbols (antenna 1 and 2 respectively) for now for testing purpose)
    s1 = np.around(np.random.randn(), 4) #random number drawn from normal distribution, rounded to 5 decimal places.
    s2 = np.around(np.random.randn(), 4) #same as above.
    Symbol_test = np.array([[s1], [s2]]) #for now, it is continous
    Symbol_test_all.append(Symbol)
    

#Evaluting performance: 
#Obtained a vector with varying noise power
start = 2
stop = 0.002
vectorLength = 20 #integer value must be an even number
stepsize = ((stop - start)/(20)) #to obtain a vector with length 10
noisepwr = np.arange(start, stop, stepsize) #Generates vector with elements used as varying noise power

snr = np.reciprocal(noisepwr) # SNR as the reciprocal of noise power
noisepwr_snr_list = [noisepwr,snr]
frame = DataFrame (noisepwr_snr_list).transpose()
frame.columns = ['noise power','signal-to-noise ratio']
frame.head()
print (frame)

#Obtaining the overall noise vector with its varying power:
#To show the noise vectors multiplied by noise powers respectively/individually
for element in noisepwr:
    #print(i, end=', ')
    noise = [element]*Noise # Generated Noise Vector (with varying noise level
    test_Noise.append(noise)

#So, i will use same sybmol[0] and Channel[0], but diff. noise power level to generate some outputs
index_ref = 1 #to define the choose index of symbol and channel to use.
test_symbol = Symbol_all[0]

for i in range(len(noisepwr)):
    y = np.add(np.dot(Channel,test_symbol),test_Noise[i]) #received signal for both time slots, both antennas
    test_Symbol_est = (np.dot(np.linalg.pinv(Channel), y).real)
    test_Symbol_est_all.append(np.reshape((test_Symbol_est), (1, dim)))


#Data Preprocessing:
#convert list to numpy array:
Symbol_all = np.array(Symbol_all)
Symbol_est_all = np.array(Symbol_est_all)
#splits the tensor array into two uniques parts. Comparing the training loss & verification loss curves will help me know underfitting or overfitting
Symobl_all_split = np.array_split(Symbol_all, 2) 
Symobl_all_validation = np.reshape((Symobl_all_split[1]), (idx,dim))
Symobl_all_training = np.reshape(Symobl_all_split[0], (idx,dim))

Symbol_est_all_split = np.array_split(Symbol_est_all, 2) #same as above, but now for the estimated symbols
Symobl_est_all_validation = np.reshape(Symbol_est_all_split[1], (idx,dim))
Symobl_est_all_training = np.reshape(Symbol_est_all_split[0], (idx,dim))


#using the same datasets generated above, I tried to:
#Train and fit the model for: 1. Autoencoders (Sparse Autoencoders), 2. Non-Spare Autoencoder 3. Supervised learning 4. Convolutional NN
#And then graphically compare the results for MSE vs SNR

#NOTE: while implementing AE through Denoisng, i am able to reconstruct our input while simultaneously eliminating any noise present within it!

#A. Supervised Learning 
def Alamouti_Scheme_A():
    global DeepNN, mean_sqaure_error_A, model
    #Building the network: Setting up layers, activation functions, optimizers, and other metrics.
    model = Sequential()
    model.add(Dense(2, init = 'random_uniform',activation='relu', input_shape =(dim,)))#first layer #I used dense layering for now here
    model.add(Dense(3 , init = 'uniform', activation='relu'))# Hidden layer
    model.add(Dense(3, init = 'random_uniform', activation='sigmoid'))#Hidden layer,
    model.add(Dense(2, init = 'uniform', activation='linear',  input_shape = (dim,)))  #Output layer,
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])
     #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(model.layers):
         layer.name = 'layer_' + str(i)
     #train the model now:
    DeepNN = model.fit(Symobl_est_all_training, Symobl_all_training, validation_data = (Symobl_est_all_validation, Symobl_all_validation), epochs=epoch, batch_size = bs, verbose= 1)
    model.summary() #obtain the summary of the network model
    #More to visualization: show the sequential layers layers
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='DeepNN.png')
    from IPython.display import Image
    Image(retina=True, filename='DeepNN.png') #saves the picture inot the folder-.py collocation
   
    pred_test_Symbol_est_all = []
    
    for i in range(len(noisepwr)):
        
        pred = model.predict(test_Symbol_est_all[i])
        pred_test_Symbol_est_all.append(pred)
        mse_A = (np.mean(test_symbol - pred_test_Symbol_est_all[i])**2) #(mean((actual-predictions)**2))
        #mse_A = (np.square(np.subtract(test_symbol, pred_test_Symbol_est_all[i]))).mean
        mean_sqaure_error_A.append(mse_A)
                    
Alamouti_Scheme_A()      


#B. Autoencoders (Non-Sparsity) #Here,typically the hidden layer is learning an approximation of PCA (principal component analysis)'
def Alamouti_Scheme_B():
    global autoencoder_wo, mean_sqaure_error_B
    #dataset dimension restructuring:
    input_shape = output_shape = 4 #input_shape or number of number of Autoencder input
    Symobl_all_validation_B = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_B = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    
    Symobl_est_all_validation_B = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_B = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))
    
    # Autoencoder Network design
    encoding_dim = 2  #no. of nodes in the bottle-neck layer
    
    #Building the network: Setting up layers, activation functions, optimizers, and other metrics.
    model = Sequential()
    model.add(Dense(4, init = 'random_uniform',activation='relu', input_shape =(input_shape,)))#first layer #I used dense layering for now here
    model.add(Dense(3 , init = 'random_uniform', activation='relu'))# Hidden layer
    model.add(Dense(encoding_dim, init = 'random_uniform', activation='relu'))#Hidden layer,
    model.add(Dense(3 , init = 'random_uniform', activation='relu'))# Hidden layer
    model.add(Dense(4, init = 'random_uniform', activation='sigmoid',  input_shape = (input_shape,)))  #Output layer,

    #model configuration, loss fuunction specificiation and training:
    model.compile(optimizer = 'adam', loss = 'mse',  metrics = ['acc'])
    autoencoder_wo = model.fit(Symobl_est_all_training_B, Symobl_all_training_B, epochs= epoch, batch_size=bs, shuffle=True, validation_data=(Symobl_est_all_validation_B, Symobl_all_validation_B))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
    #obtain the summary of the network model
    model.summary()  
    
    #More to visualization: show the sequential layers layers
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='autoencoder_wo.png')
    from IPython.display import Image
    Image(retina=True, filename='autoencoder_wo.png') #saves the picture inot the folder-.py collocation
           
    pred_test_Symbol_est_allB = []
    kk = []
    iterationfactor = int((len(noisepwr))/2)
    
    #list manipulation: to allow the expect dimenisonal to be predicted by the trained model
    #selects every 2 vectors, concatenates them as 1 vectors, and appends them onto a new list to form a new vector/array
    rr = 0
    for j in range(iterationfactor):
        qq = np.concatenate((test_Symbol_est_all[rr], test_Symbol_est_all[rr+1]), axis=1) #since the szie of imput of the Autoencoder model is 4, i had to concatentate 2 vectors to make up to 4 elements.
        kk.append(qq)
        rr = rr+2
             
    for i in range(iterationfactor):   
        pred = model.predict(kk[i]) #the new data is predicted after training the model
        pred_test_Symbol_est_allB.append(pred) # then appended, yet still with dimension 1by4
     
    dd = ((np.array(pred_test_Symbol_est_allB)).flatten()) #convert list to array, and then flattened.
    #Note: To compute the MSE, I compare the testSymbol (which is 1by2) with pred_test_Symbol_est_allB (which should be 1by2 also)
    #I flattened the array to enable me selective slice the long list in 2s which should be fit to be compared with the testSymbol array/list
            
    pred_test_Symbol_est__allB = []#assigns an empty list which will hold the slices from the long list, dd
        
    for m in range(int(len(dd)*0.5)):
        pred_test_Symbol_est__allB.append(dd[:2]) #appends on the first 2 elements
        dd = np.delete(dd, (np.arange(2))) #and the deletes the 2 elements after appending. This is done iteratively till completed.
    mean_sqaure_error_B = []
    for i in range(len(pred_test_Symbol_est__allB)):
      
        mse_B = (np.mean(test_symbol - pred_test_Symbol_est__allB[i])**2) #computes the mean square error
        mean_sqaure_error_B.append(mse_B)

           
Alamouti_Scheme_B()
#or more detailed using this:

# encoded_symbols = encoder.predict(test_Symbol_est_all[5]) #because I want to view the output/latent representation of the testSymbol with index 5
# decoded_symbols = decoder.predict(encoded_symbols)


#C. Autoencoders with Sparsity #this is achieved by using Regularizers. A specfic activation is fired at a given time.' 
def Alamouti_Scheme_C():
    global autoencoder_w, mean_sqaure_error_C
    #dataset dimension restructuring:
    input_shape = output_shape = 4 #input_shape or number of number of Autoencder input
    encoding_dim = 6 #since sparsity is considered
    
    #load the required dataset.
    Symobl_all_validation_C = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_C = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_C = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_C = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))  
    
    
    #Building the network: Setting up layers, activation functions, optimizers, and other metrics.
    model = Sequential()
    model.add(Dense(4, init = 'random_uniform',activation='relu', input_shape =(input_shape,)))#first layer #I used dense layering for now here
    model.add(Dense(3, init = 'uniform', activation='relu'))# Hidden layer
    model.add(Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(1e-5)))#Hidden layer, l1 Regularizer, Dropouts could work too.
    model.add(Dense(3 , init = 'uniform', activation='relu'))# Hidden layer
    model.add(Dense(4, init = 'uniform', activation='sigmoid',  input_shape = (output_shape,)))  #Output layer,
    
    model.compile(optimizer = 'adam', loss = 'mse',  metrics = ['acc'])
    autoencoder_w = model.fit(Symobl_est_all_training_C, Symobl_all_training_C, epochs= epoch, batch_size=bs, shuffle=True, validation_data=(Symobl_est_all_validation_C, Symobl_all_validation_C))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
        
    #obtain the summary of the network model
    model.summary() 
    #More to visualization: show the sequential layers layers
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='autoencoder_w.png')
    from IPython.display import Image
    Image(retina=True, filename='autoencoder_w.png') #saves the picture inot the folder-.py collocation
    
    
    pred_test_Symbol_est_allC = []
    kk = []
    iterationfactor = int((len(noisepwr))/2)
    
    #list manipulation: to allow the expect dimenisonal to be predicted by the trained model
    #selects every 2 vectors, concatenates them as 1 vectors, and appends them onto a new list to form a new vector/array
    rr = 0
    for j in range(iterationfactor):
        qq = np.concatenate((test_Symbol_est_all[rr], test_Symbol_est_all[rr+1]), axis=1) #since the szie of imput of the Autoencoder model is 4, i had to concatentate 2 vectors to make up to 4 elements.
        kk.append(qq)
        rr = rr+2
             
    for i in range(iterationfactor):   
        
        pred = model.predict(kk[i]) #the new data is predicted after training the model
        pred_test_Symbol_est_allC.append(pred) # then appended, yet still with dimension 1by4
     
    dd = ((np.array(pred_test_Symbol_est_allC)).flatten()) #convert list to array, and then flattened.
    #Note: To compute the MSE, I compare the testSymbol (which is 1by2) with pred_test_Symbol_est_allB (which should be 1by2 also)
    #I flattened the array to enable me selective slice the long list in 2s which should be fit to be compared with the testSymbol array/list
            
    pred_test_Symbol_est__allC = []#assigns an empty list which will hold the slices from the long list, dd
        
    for m in range(int(len(dd)*0.5)):
        pred_test_Symbol_est__allC.append(dd[:2]) #appends on the first 2 elements
        dd = np.delete(dd, (np.arange(2))) #and the deletes the 2 elements after appending. This is done iteratively till completed.
    
    mean_sqaure_error_C = []
    for i in range(len(pred_test_Symbol_est__allC)):
      
        mse_C = (np.mean(test_symbol - pred_test_Symbol_est__allC[i])**2) #computes the mean square error
        mean_sqaure_error_C.append(mse_C)
    
Alamouti_Scheme_C()   
    
#D - Variational Autoencoder 
def Alamouti_Scheme_D():
    global variational_autoencoder, mean_sqaure_error_D
    original_dim = input_shape = 4
    original_dim_b = 4
    intermediate_dim = 3
    latent_dim = 2
    epsilon_std = 1.0
    
    Symobl_all_validation_D = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_D = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_D = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_D = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))
    
    
    class KLDivergenceLayer(Layer): #kullback_leibler_divergence regularizer - used a reparameterizer
    
        #Identities transform layer that adds KL divergence to the final model loss.
        def __init__(self, *args, **kwargs):
            self.is_placeholder = True
            super(KLDivergenceLayer, self).__init__(*args, **kwargs)
    
        def call(self, inputs):
            mu, log_var = inputs #mean, variance.
            kl_batch = (-0.5) * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis = -1)
            self.add_loss(K.mean(kl_batch), inputs=inputs)
    
            return inputs
    #define the input (encoder) layers
    layerA = Input(shape=(original_dim,))
    layerB = Dense(original_dim_b, activation='relu')(layerA)
    layerC = Dense(intermediate_dim, activation='relu')(layerB) #input to latent/Bottle neck
    
    #defines the latent layers
    z_mu = Dense(latent_dim, name = 'latent_mu')(layerC) #mean of the encoded sample
    z_log_var = Dense(latent_dim, name = 'latent_var')(layerC) #standard deviation of the encoded samples
    
    
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
    eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(layerA)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])
    
    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim_b, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')])
    
    decoded_preds = decoder(z) #decoded predictions or representations
    
    vae = Model(inputs=[layerA, eps], outputs=decoded_preds)
    vae.compile(optimizer = 'adam', loss = 'mse',  metrics = ['acc'])
    
    variational_autoencoder = vae.fit(Symobl_est_all_training_D, Symobl_all_training_D, shuffle=True, epochs=epoch, batch_size=bs, validation_data=(Symobl_est_all_validation_D, Symobl_all_validation_D))
    vae.summary()
 
    #More to visualization: show the sequential layers layers
    plot_model(vae, show_shapes=True, show_layer_names=True, to_file='variational_autoencoder.png')
    from IPython.display import Image
    Image(retina=True, filename='variational_autoencoder.png') #saves the picture inot the folder-.py collocation
    
    pred_test_Symbol_est_allD = []
    kk = []
    iterationfactor = int((len(noisepwr))/2)
    
    #list manipulation: to allow the expect dimenisonal to be predicted by the trained model
    #selects every 2 vectors, concatenates them as 1 vectors, and appends them onto a new list to form a new vector/array
    rr = 0
    for j in range(iterationfactor):
        qq = np.concatenate((test_Symbol_est_all[rr], test_Symbol_est_all[rr+1]), axis=1) #since the szie of imput of the Autoencoder model is 4, i had to concatentate 2 vectors to make up to 4 elements.
        kk.append(qq)
        rr = rr+2
             
    for i in range(iterationfactor):   
        
        pred = vae.predict(kk[i]) #the new data is predicted after training the model
        pred_test_Symbol_est_allD.append(pred) # then appended, yet still with dimension 1by4
     
    dd = ((np.array(pred_test_Symbol_est_allD)).flatten()) #convert list to array, and then flattened.
    #Note: To compute the MSE, I compare the testSymbol (which is 1by2) with pred_test_Symbol_est_allB (which should be 1by2 also)
    #I flattened the array to enable me selective slice the long list in 2s which should be fit to be compared with the testSymbol array/list
            
    pred_test_Symbol_est__allD = []#assigns an empty list which will hold the slices from the long list, dd
        
    for m in range(int(len(dd)*0.5)):
        pred_test_Symbol_est__allD.append(dd[:2]) #appends on the first 2 elements
        dd = np.delete(dd, (np.arange(2))) #and the deletes the 2 elements after appending. This is done iteratively till completed.
    
    mean_sqaure_error_D = []
    for i in range(len(pred_test_Symbol_est__allD)):
      
        mse_D = (np.mean(test_symbol - pred_test_Symbol_est__allD[i])**2) #computes the mean square error
        mean_sqaure_error_D.append(mse_D)
     
    
    
Alamouti_Scheme_D()

#plots mean_squared_error over snr:
plt.plot(snr, mean_sqaure_error_A)
plt.plot(snr, mean_sqaure_error_B)
plt.plot(snr, mean_sqaure_error_C)
plt.plot(snr, mean_sqaure_error_D)
plt.title('Graph of MSE with varying SNR (after prediction)')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.ylabel('mse')
plt.xlabel('snr')
plt.grid(b=None, which='major', axis='both')
plt.show()

#plots loss (b oth training and validation) over epoch:
plt.plot(DeepNN.history['loss'])
plt.plot(autoencoder_wo.history['loss'])
plt.plot(autoencoder_w.history['loss'])
plt.plot(variational_autoencoder.history['loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


plt.plot(DeepNN.history['val_loss'])
plt.plot(autoencoder_wo.history['val_loss'])
plt.plot(autoencoder_w.history['val_loss'])
plt.plot(variational_autoencoder.history['val_loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Validation Loss')
plt.xlabel('Epoch')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


# Accuracy
#plots loss (b oth training and validation) over epoch:
plt.plot(DeepNN.history['acc'])
plt.plot(autoencoder_wo.history['acc'])
plt.plot(autoencoder_w.history['acc'])
plt.plot(variational_autoencoder.history['acc'])
plt.title('Graph of final Performance Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


plt.plot(DeepNN.history['val_acc'])
plt.plot(autoencoder_wo.history['val_acc'])
plt.plot(autoencoder_w.history['val_acc'])
plt.plot(variational_autoencoder.history['val_acc'])
plt.title('Graph of final Performance Training Accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()




