"""
Created on Sun Dec 2 11:28:29 2020
'So, I tried to evaluate the performance of several flavours of Autoencoders on Alamouti Coded Symbol (continous-valued data)'
'The following graphically result was performed on same dataset, epoch value and batchsize'
#Evaluation was based on: MSE vs. SNR & Loss vs. Epoch Value.
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
from keras.models import Sequential, Model
from keras.layers import Dense,Input, Dropout, Activation, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Lambda, Layer, Add, Multiply
from keras.utils import plot_model
from pandas import DataFrame


#Variable Declaration and Assignment
tx_antenna = 2 #number of tx_antennas
rx_antenna = 2 #number of rx_antennas
dim = tx_antenna  #NB: dim has to be a value which is a factor of 'noOfSamples'.
noOfSamples = int(120000) #number of sample data generated in the for-loop. It determines the 3 dimension of our matrix or vector.

if noOfSamples % 2==0:     #just a control measure to be sure ##noOfSamples is divisible be 2 always.
    noOfSamples = noOfSamples
else:
    noOfSamples = noOfSamples + 1

idx = int(noOfSamples/2) # the loop index will be the number of the splitted parts of the 3D tensor/array
epoch = 50
layer1node = 2 # number of nodes in first layer or the length of the
layer2anode = 4 # number of nodes in second layer (hidden)
layer2bnode = 4 # number of nodes in third layer (hidden)
layer3node = 2 # number of nodes in fourth layer
#to be able to reeuse the content of the variable, we set them as global variables: 
#Recall, the varibales are within a function, so I got to say they are global variables.
global Symbol_est_all_split, Symobl_all_split, Symbol_test_all, snr

snr = None
Symobl_all_split = None
Symbol_est_all_split = None
test_Symbol_est_all = None
DeepNN = None
mean_sqaure_error_A = None
mean_sqaure_error_B = None
autoencoder_wo = None
autoencoder_w =None
variational_autoencoder = None

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
    global DeepNN, mean_sqaure_error_A
    #Building the network: Setting up layers, activation functions, optimizers, and other metrics.
    model = Sequential()
    model.add(Dense(layer1node, init = 'random_uniform',activation='relu', input_shape =(dim,)))#first layer #I used dense layering for now here
    model.add(Dense(layer2anode , init = 'uniform', activation='sigmoid'))# Hidden layer
    model.add(Dense(layer2bnode, init = 'random_uniform', activation='relu'))#Hidden layer, 
    model.add(Dense(layer3node, init = 'uniform', activation='linear',  input_shape = (dim,)))  #Output layer,
    model.compile(optimizer = 'adam', loss = 'mse')
     #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(model.layers):
         layer.name = 'layer_' + str(i)
     #train the model now:
    DeepNN = model.fit(Symobl_est_all_training, Symobl_all_training, validation_data = (Symobl_est_all_validation, Symobl_all_validation), epochs=epoch, batch_size =  dim, verbose= 1)
    model.summary() #obtain the summary of the network model
    #More to visualization: show the sequential layers layers
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='DeepNN.png')
    from IPython.display import Image
    Image(retina=True, filename='DeepNN.png') #saves the picture inot the folder-.py collocation
   
    pred_test_Symbol_est_all = []
    
    for i in range(len(noisepwr)):
        
        pred = model.predict(test_Symbol_est_all[i])
        pred_test_Symbol_est_all.append(pred)
        mse_A = (np.mean(test_symbol - pred_test_Symbol_est_all[i])**2)
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
    encoding_dim = 3  #no. of nodes in the bottle-neck layer
    model = Sequential()
    input_symbol = keras.Input(shape=(input_shape,)) #input of the network
    encoder_layer2 = layers.Dense(input_shape, activation='relu')(input_symbol)
    
    encoded = layers.Dense(encoding_dim, activation='relu')(encoder_layer2) #the encoded representation of the input
    
    decoder_layer2 = layers.Dense(output_shape, activation='relu')(encoded)
    decoded = layers.Dense(output_shape, activation='sigmoid')(decoder_layer2) #lossy reconstruction of the input
    
    # model that maps an input to its reconstruction
    autoencoder = keras.Model(input_symbol, decoded)
    ## This model maps an input to its encoded representation
    encoder = keras.Model(input_symbol, encoded)
    #to showed the encoded input (with dimension encoding_dim):
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model. that is the layer that starts the decoder model
    decoder_layer = autoencoder.layers[3]
    #to create a decoder using the last layer of the autoencoder model@
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input)) 
    
    #model configuration, loss fuunction specificiation and training:
    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    autoencoder_wo = autoencoder.fit(Symobl_est_all_training_B, Symobl_all_training_B, epochs= epoch, batch_size=dim, shuffle=True, validation_data=(Symobl_est_all_validation_B, Symobl_all_validation_B))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(model.layers):
        layer.name = 'layer_' + str(i)
    #obtain the summary of the network model
    autoencoder.summary()  
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
        pred = autoencoder.predict(kk[i]) #the new data is predicted after training the model
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

# encoded_symbols = encoder.predict(symbol_test)
# decoded_symbols = decoder.predict(encoded_imgs)


#C. Autoencoders with Sparsity #this is achieved by using Regularizers. A specfic activation is fired at a given time.' 
def Alamouti_Scheme_C():
    global autoencoder_w, pred_test_Symbol_est_allC
    #dataset dimension restructuring:
    input_shape = output_shape = 6 #input_shape or number of number of Autoencder input
    encoding_dim = 4
    
    #load the required dataset.
    Symobl_all_validation_C = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_C = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_C = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_C = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))  
    
    autoencoder_withSparsity = Sequential() 
    input_symbol = keras.Input(shape=(input_shape,)) #input of the network
    encoder_layer2 = layers.Dense(5, activation='relu')(input_symbol)
    
    encoded = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(encoder_layer2)
    decoder_layer2 = layers.Dense(5, activation='relu')(encoded)
    decoded = layers.Dense(output_shape, activation='sigmoid')(decoder_layer2) #lossy reconstruction of the input
    
       
    autoencoder_withSparsity.compile(optimizer = 'adam', loss = 'mse')
    autoencoder_w = autoencoder_withSparsity.fit(Symobl_est_all_training_C, Symobl_all_training_C, epochs= epoch, batch_size=dim, shuffle=True, validation_data=(Symobl_est_all_validation_C, Symobl_all_validation_C))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(autoencoder_withSparsity.layers):
        layer.name = 'layer_' + str(i)
        
    #obtain the summary of the network model
    autoencoder_withSparsity.summary() 
    #More to visualization: show the sequential layers layers
    plot_model(autoencoder_withSparsity, show_shapes=True, show_layer_names=True, to_file='autoencoder_w.png')
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
    
    #to obtain the outputs of the encoder and decoder respectively, with possibilty of the array shapes:
    #NOTE: You have to generate a whole new set of data to test the trained model.

    
#D - Variational Autoencoder 
def Alamouti_Scheme_D():
    global variational_autoencoder
#A variational autoencoder (VAE) provides a probabilistic manner for describing an observation in latent space
    original_dim = input_shape = 6
    original_dim_b = 6
    intermediate_dim = 4
    latent_dim = 3
    epsilon_std = 1.0
    
    Symobl_all_validation_D = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_D = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_D = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_D = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))
    
    
    class KLDivergenceLayer(Layer): #kullback_leibler_divergence regularizer
    
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
    x = Input(shape=(original_dim,))
    w = Dense(original_dim_b, activation='relu')(x)
    h = Dense(intermediate_dim, activation='relu')(w)
    
    #defines the latent layers
    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
    eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])
    
    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim_b, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')])
    x_pred = decoder(z)
    
    vae = Model(inputs=[x, eps], outputs=x_pred)
    #vae.compile(optimizer='adam', loss=nll)
    vae.compile(optimizer = 'adam', loss = 'mse')
    
    x_train = Symobl_est_all_training_D
    x__train =  Symobl_all_training_D
    x_test = Symobl_est_all_validation_D
    x__test = Symobl_all_validation_D
    
    variational_autoencoder = vae.fit(Symobl_est_all_training_D, Symobl_all_training_D, shuffle=True, epochs=epoch, batch_size=dim, validation_data=(Symobl_est_all_validation_D, Symobl_all_validation_D))
    vae.summary()
    #More to visualization: show the sequential layers layers
    plot_model(vae, show_shapes=True, show_layer_names=True, to_file='variational_autoencoder.png')
    from IPython.display import Image
    Image(retina=True, filename='variational_autoencoder.png') #saves the picture inot the folder-.py collocation
    
    encoder = Model(x, z_mu)

Alamouti_Scheme_D()

#Convolutional Autoencoders:
def Alamouti_Scheme_E():  
    
    input_shape = output_shape = 24  #input_shape or number of number of Autoencder input

    Symobl_all_validation_E = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_E = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_E = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_E = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))
  
    autoencoder_withConv2D = Sequential() 
    #Encoder:
    input_symbol = keras.Input(shape=(24, 24, 1))
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_symbol)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder_withConv2D = keras.Model(input_symbol, decoded)
    
    autoencoder_withConv2D.compile(optimizer='adam', loss='mse')
    
    autoencoder_withConv2D.fit(Symobl_est_all_training_E, Symobl_all_training_E, epochs= epoch, batch_size=dim, shuffle=True, validation_data=(Symobl_est_all_validation_E, Symobl_all_validation_E))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(autoencoder_withConv2D.layers):
        layer.name = 'layer_' + str(i)
        
      #obtain the summary of the network model
    autoencoder_withConv2D.summary() 

Alamouti_Scheme_E()  

#plots mean_squared_error over snr:
plt.plot(snr, mean_sqaure_error_A)
plt.plot(snr, mean_sqaure_error_B)
plt.title('Graph of MSE with varying SNR (after prediction)')
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
plt.xlabel('No. of Epoch')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


plt.plot(DeepNN.history['val_loss'])
plt.plot(autoencoder_wo.history['val_loss'])
plt.plot(autoencoder_w.history['val_loss'])
plt.plot(variational_autoencoder.history['val_loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Validation Loss')
plt.xlabel('No. of Epoch')
plt.legend(['DeepNN', 'autoencoder_w/o', 'autoencoder_w', 'v_autoencoder'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


