from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Lambda, Flatten, Reshape, MaxPooling2D
from keras.models import Model
from keras.losses import mean_squared_error, binary_crossentropy, kullback_leibler_divergence
from keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, scipy
from sklearn.model_selection import train_test_split

from phenom_to_df import *

#Get the data
df = csv_to_df('sm_data/sm_10fb/test/')
df = df.fillna(0)
variables = [entry for entry in df.columns if entry[0] == 'E'] + [entry for entry in df.columns if entry[0:2] == 'pt'] + [entry for entry in df.columns if entry[0:2] == 'et'] + [entry for entry in df.columns if entry[0:2] == 'ph']

#Now we one_hot encode the data to make sure we have a class definition
df = df[['process_ID']+variables]
one_hot = pd.get_dummies(df['process_ID'])
processes = one_hot.columns
df.drop('process_ID', axis = 'columns', inplace = True)
df = pd.concat([df, one_hot], sort = False, axis = 1)

#We have created a df of our values and some kind of class label
#We should normalize
x = df[variables].values
x_scaled = StandardScaler().fit_transform(x)
df[variables] = x_scaled

x_train, x_test, y_train, y_test = train_test_split(df[variables].values,
						    df[processes].values, 
                                                    shuffle = True,
                                                    random_state = 42,
                                                    test_size = 0.1)

#Reshape the data. This will have to be done explicitly until you think of a clever way of doing it
x_train = x_train.reshape((x_train.shape[0], 4, 2, 7))
x_test = x_test.reshape((x_test.shape[0], 4, 2, 7))

def sampling(args):
    z_mean, z_log_var=args
    epsilon=tf.random.uniform(shape=(K.shape(z_mean)[0], latent_dim))
    #Using backend (K) here ensures that our function works over tf and keras
    #I don't think it's actually necessary but I will keep it in
    return z_mean+K.exp(z_log_var/2)*epsilon


#VAE Model
original_dim = x_train.shape[0]
input_shape = x_train.shape[1:]
latent_dim = 4
intermediate_dim = 50
kernel_max_norm = 1000.
act_fun = 'relu'
epsilon_std = 1.
filters = 32
kernel_size = (3, 3)
strides = (1, 1)


def make_model(optimizer = 'rmsprop', filters = filters, intermediate_dim = intermediate_dim, act_fun = act_fun, 
               kernel_size = kernel_size, strides = strides):
    
    def sampling(args):
        z_mean, z_log_var=args
        epsilon=tf.random.uniform(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean+K.exp(z_log_var/2)*epsilon
    
    #Define some losses
    def kl_loss(y_true, y_pred):
        kl_loss = 1 + z_var - K.square(z_mean) - K.exp(z_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        #kl_loss = K.mean(kl_loss)
        return kl_loss

    def recon_loss(y_true, y_pred):
        recon_loss = mean_squared_error(K.flatten(y_true), K.flatten(y_pred))
        recon_loss *= shape[0]*shape[1]*shape[2]
        #recon_loss = K.mean(recon_loss)
        return recon_loss

    def my_vae_loss(y_true, y_pred):
        recon = recon_loss(y_true, y_pred) 
        kl = kl_loss(y_true, y_pred)
        vae_loss = recon + kl
        return vae_loss

    
    #Layers
    x_input = Input(shape = input_shape, name = 'Input')
    conv_1 = Conv2D(filters = filters, 
                activation = act_fun,
                kernel_size = kernel_size,
                strides = strides, 
                padding = 'same', 
                name = 'conv_1')(x_input)
    pool_1 = MaxPooling2D(pool_size = pool_size,
                          strides = 2, 
                          name = 'max_pool')(conv_1)
    conv_2 = Conv2D(filters = filters, 
                activation = act_fun, 
                kernel_size = kernel_size, 
                strides = strides, 
                padding = 'same', 
                name = 'conv_2')(pool_1)
    #Shape info needed to inform the dense layer in the decoder
    flatten = Flatten()(conv_2)#(pool_1)
    cnn = Model(x_input, flatten, name = 'cnn')
    cnn.summary()

    shape = input_shape
    
    #Encoding
    dense_encoder_1 = Dense(intermediate_dim,
                            activation = act_fun,
                            name = 'dense_encoder_1')(flatten)
    dense_encoder_2 = Dense(16,
                            activation = act_fun,
                            name = 'dense_encoder_2')(dense_encoder_1)
    z_mean = Dense(latent_dim, name = 'z_mean')(dense_encoder_2)
    z_var = Dense(latent_dim, name = 'z_var')(dense_encoder_2)

    z = Lambda(sampling, output_shape = (latent_dim, ), name = 'sampling')([z_mean, z_var])

    encoder = Model(x_input, [z_mean, z_var, z], name = 'encoder')
    encoder.summary()
    
    #Decoding
    dense_shape = shape[0]*shape[1]*shape[2]
    decoder_input = Input(shape = (latent_dim, ), name = 'decoder_input')
    dense_decoder_1 = Dense(dense_shape, 
                            activation = act_fun, 
                            name = 'dense_decoder_1')(decoder_input)
    reshape = Reshape((shape[0], shape[1], shape[2]))(dense_decoder_1)

    deconv_1 = Conv2DTranspose(filters = filters,
                               kernel_size = kernel_size, 
                               activation = act_fun, 
                               strides = strides, 
                               padding = 'same', 
                               name = 'deconv_1')(reshape)
    deconv_2 = Conv2DTranspose(filters = 20,
                               kernel_size = kernel_size, 
                               activation = act_fun, 
                               strides = strides, 
                               padding = 'same', 
                               name = 'deconv_2')(deconv_1)

    output = Conv2DTranspose(filters = 4, 
                             kernel_size = kernel_size, 
                             #activation = 'sigmoid', 
                             padding = 'same', 
                             strides = strides, 
                             name = 'decoder_output')(deconv_2)

    decoder = Model(decoder_input, output, name = 'decoder')
    decoder.summary()
    
    #Build the VAE model
    outputs = decoder(encoder(x_input)[2])
    vae = Model(x_input, outputs, name = 'vae')
    
    vae.compile(optimizer = optimizer, loss = my_vae_loss, metrics = [kl_loss, recon_loss])
    vae.summary()
    
    return vae

#Fit the model
epochs = 100
batch_size = 1000

vae = make_model()

history = vae.fit(x = x_train,
                  y = x_train,
                  validation_data = (x_train, x_train),
                  epochs = epochs,
                  batch_size = batch_size,
                  verbose = 2)

vae.save('cnn-vae_model.h5')

y_pred = vae.predict(x_train.reshape(x_train.shape[0], 1, 14, 4))
x_train_reshape = x_train.reshape(x_train.shape[0], 56)
x_train_df = pd.DataFrame(x_train_reshape, columns = variables)
x_train_E = np.array(x_train_df[E])
x_train_pt = np.array(x_train_df[pt])
x_train_phi = np.array(x_train_df[phi])
x_train_eta = np.array(x_train_df[eta])
y_pred_reshape = y_pred.reshape(y_pred.shape[0], 56)
y_pred_df = pd.DataFrame(y_pred_reshape, columns = variables)
y_pred_E = np.array(y_pred_df[E])
y_pred_pt = np.array(y_pred_df[pt])
y_pred_phi = np.array(y_pred_df[phi])
y_pred_eta = np.array(y_pred_df[eta])
x_train_flat = x_train_reshape.flatten()
y_pred_flat = y_pred_reshape.flatten()

n, bins, patches = plt.hist(x_train_E.flatten()[:100000], 1000)
plt.figure(figsize=(10,7))
plt.xlim(0, 10)
plt.xlabel('Scaled Input')
plt.ylabel('Frequency')
plt.title('Energy Data Histogram')
plt.hist(x_train_E.flatten()[:100000], bins = bins, color='blue', log=True, label='Input Data', alpha=0.5)
plt.hist(y_pred_E.flatten()[:100000], bins = bins, color='orange', log=True, label='Prediction', alpha=0.5)
plt.legend()
plt.show()

n, bins, patches = plt.hist(x_train_pt.flatten()[:100000], 1000)
plt.figure(figsize=(10,7))
plt.xlim(0, 10)
plt.xlabel('Scaled Input')
plt.ylabel('Frequency')
plt.title('p_T Data Histogram')
plt.hist(x_train_pt.flatten()[:100000], bins = bins, color='blue', log=True, label='Input Data', alpha=0.5)
plt.hist(y_pred_pt.flatten()[:100000], bins = bins, color='orange', log=True, label='Prediction', alpha=0.5)
plt.legend()
plt.show()

n, bins, patches = plt.hist(x_train_phi.flatten()[:100000], 1000)
plt.figure(figsize=(10,7))
plt.xlim(0, 10)
plt.xlabel('Scaled Input')
plt.ylabel('Frequency')
plt.title('Input Phi Data Histogram')
plt.hist(x_train_phi.flatten()[:100000], bins = bins, color='blue', log=True, label='Input Data', alpha=0.5)
plt.hist(y_pred_phi.flatten()[:100000], bins = bins, color='orange', log=True, label='Prediction', alpha=0.5)
plt.legend()
plt.show()

n, bins, patches = plt.hist(x_train_eta.flatten()[:100000], 1000)
plt.figure(figsize=(10,7))
plt.xlim(0, 10)
plt.xlabel('Scaled Input')
plt.ylabel('Frequency')
plt.title('Input Eta Data Histogram')
plt.hist(x_train_eta.flatten()[:100000], bins = bins, color='blue', log=True, label='Input Data', alpha=0.5)
plt.hist(y_pred_eta.flatten()[:100000], bins = bins, color='orange', log=True, label='Prediction', alpha=0.5)
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['kl_loss'])
plt.title('Kullback-Liebler Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()

plt.plot(history.history['recon_loss'])
plt.title('Reconstruction (MSE) Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()
