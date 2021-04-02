import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import keras
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split #split dataset into training and test data sets
from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense,Input, Dropout, Activation, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Lambda, Layer, Add, Multiply
from keras.utils import plot_model


#Variable Declaration and Assignment
tx_antenna = 2 #number of tx_antennas
rx_antenna = 2 #number of rx_antennas
dim = tx_antenna #NB: dim has to be a value which is a factor of 'noOfSamples'.
noOfSamples = int(1200) #number of sample data generated in the for-loop. It determines the 3 dimension of our matrix or vector.

if noOfSamples % 2==0:     #just a control measure to be sure ##noOfSamples is divisible be 2 always.
    noOfSamples = noOfSamples
else:
    noOfSamples = noOfSamples + 1

idx = int(noOfSamples/2) # the loop index will be the number of the splitted parts of the 3D tensor/array
epoch = 100
layer1node = 2 # number of nodes in first layer or the length of the
layer2anode = 4 # number of nodes in second layer (hidden)
layer2bnode = 4 # number of nodes in third layer (hidden)
layer3node = 2 # number of nodes in fourth layer
#to be able to reeuse the content of the variable, we set them as global variables: 
#Recall, the varibales are within a function, so I got to say they are global variables.
global Symbol_est_all_split, Symobl_all_split, Symbol_test_all


Symobl_all_split = None
Symbol_est_all_split = None
Symbol_test_all = None
DeepNN = None
autoencoder_wo = None
autoencoder_w =None
variational_autoencoder = None

X_all = []
Symbol_all = [] #all transmitted symbol
Symbol_est_all = [] #all recieved or estimated symbols
Symbol_test_all = [] #all test symbols for evaluation of trained model


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
    global DeepNN
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
    
    # pred = model.predict(Symbol_test)
    # mse = np.sqrt(metrics.mean_squared_error(pred,Symbol_test)) 
Alamouti_Scheme_A()    


#B. Autoencoders (Non-Sparsity) #Here,typically the hidden layer is learning an approximation of PCA (principal component analysis)'
def Alamouti_Scheme_B():
    global autoencoder_wo
    #dataset dimension restructuring:
    input_shape = output_shape = 6 #input_shape or number of number of Autoencder input
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
    
Alamouti_Scheme_B()
 
#to predict using the trained model:
#decoded_symbols = autoencoder.predict(symbol_test)
#or more detailed using this:

# encoded_symbols = encoder.predict(symbol_test)
# decoded_symbols = decoder.predict(encoded_imgs)


#C. Autoencoders with Sparsity #this is achieved by using Regularizers. A specfic activation is fired at a given time.' 
def Alamouti_Scheme_C():
    global autoencoder_w
    #dataset dimension restructuring:
    input_shape = output_shape = 6 #input_shape or number of number of Autoencder input
    encoding_dim = 3
    
    Symobl_all_validation_C = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_C = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_C = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_C = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))  
    autoencoder_withSparsity = Sequential() 
    input_symbol = keras.Input(shape=(input_shape,)) #input of the network
    encoder_layer2 = layers.Dense(4, activation='relu')(input_symbol)
    
    encoded = layers.Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(encoder_layer2)
    
    decoder_layer2 = layers.Dense(4, activation='relu')(encoded)
    decoded = layers.Dense(output_shape, activation='sigmoid')(decoder_layer2) #lossy reconstruction of the input
    
    # model that maps an input to its reconstruction
    autoencoder_withSparsity = keras.Model(input_symbol, decoded)
    ## This model maps an input to its encoded representation
    encoder = keras.Model(input_symbol, encoded)
    #to showed the encoded input (with dimension encoding_dim):
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model. that is the layer that starts the decoder model
    decoder_layer = autoencoder_withSparsity.layers[3]
    #to create a decoder using the last layer of the autoencoder model@
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input)) 
       
    autoencoder_withSparsity.compile(optimizer = 'adam', loss = 'mse')
    autoencoder_w = autoencoder_withSparsity.fit(Symobl_est_all_training_C, Symobl_all_training_C, epochs= epoch, batch_size=dim, shuffle=True, validation_data=(Symobl_est_all_validation_C, Symobl_all_validation_C))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(autoencoder_withSparsity.layers):
        layer.name = 'layer_' + str(i)
        
    #obtain the summary of the network model
    autoencoder_withSparsity.summary() 
    
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
        Dense(original_dim, activation='sigmoid')
    ])
    
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
    encoder = Model(x, z_mu)

Alamouti_Scheme_D()


#Convolutional Autoencoders:
def Alamouti_Scheme_E():  
    
    input_shape = output_shape = 24  #input_shape or number of number of Autoencder input

    Symobl_all_validation_D = np.reshape((Symobl_all_split[1]), (int(noOfSamples/input_shape),input_shape))
    Symobl_all_training_D = np.reshape(Symobl_all_split[0], (int(noOfSamples/input_shape),input_shape))
    
    Symobl_est_all_validation_D = np.reshape(Symbol_est_all_split[1], (int(noOfSamples/input_shape),input_shape))
    Symobl_est_all_training_D = np.reshape(Symbol_est_all_split[0],  (int(noOfSamples/input_shape),input_shape))
  
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
    
    autoencoder_withConv2D.fit(Symobl_est_all_training_D, Symobl_all_training_D, epochs= epoch, batch_size=dim, shuffle=True, validation_data=(Symobl_est_all_validation_D, Symobl_all_validation_D))
    
    #to rename layer for easier localization from model somaary:
    for i, layer in enumerate(autoencoder_withConv2D.layers):
        layer.name = 'layer_' + str(i)
        
      #obtain the summary of the network model
    autoencoder_withConv2D.summary() 



Alamouti_Scheme_E()  



#*************************
    x_up = ZeroPadding2D((2, 2))(x)
    h_1 = Conv2D(16, (3, 3), padding= 'same', activation = 'relu')(x_up)
    p_1 = MaxPooling2D((2, 2))(h_1)
    h_2 = Conv2D(8, (3, 3), padding= 'same', activation = 'relu')(p_1)
    p_2 = MaxPooling2D((2, 2))(h_2)
    h_3 = Conv2D(8, (3, 3), padding= 'same',activation = 'relu')(p_2)
    z = MaxPooling2D((2, 2))(h_3)
    
    #Decoder:
    h_4 = Conv2DTranspose(8, (3, 3), padding= 'same', strides=(2, 2), activation= 'relu')(z)
    h_5 = Conv2DTranspose(16, (3, 3), padding= 'same', strides=(2, 2), activation= 'relu')(h_4)
    y_up = Conv2DTranspose(1, (3, 3), padding= 'same', strides=(2, 2), activation= 'sigmoid')(h_5)
    y = Cropping2D((2, 2))(y_up)

    
    autoencoder_withConv2D = keras.Model(x, y)
    
###more example: convolutional autoencoder:

# creating autoencoder model
encoder_inputs = Input(shape = (24,24,1))
 
conv1 = Conv2D(16, (3,3), activation = 'relu', padding = "SAME")(encoder_inputs)
pool1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv1)
conv2 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(pool1)
pool2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv2)
flat = Flatten()(pool2)
 
enocder_outputs = Dense(24, activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(flat)
 
#upsampling in decoder
 
dense_layer_d = Dense(7*7*24, activation = 'relu')(enocder_outputs)
output_from_d = np.reshape((7,7,32))(dense_layer_d)
conv1_1 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(output_from_d)
upsampling_1 = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(conv1_1)
upsampling_2 = Conv2DTranspose(16, 3, padding='same', activation='relu', strides=(2, 2))(upsampling_1)
decoded_outputs = Conv2DTranspose(1, 3, padding='same', activation='relu')(upsampling_2)
 
autoencoder = Model(encoder_inputs, decoded_outputs)





###The Loss Function as MSE - The Reconstruction Loss which penalizes the network for creating outputs different from the input
