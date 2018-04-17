# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:22:18 2017

@author: haidar
"""
import scipy
import scipy.signal
import numpy as np
from matplotlib import pyplot as plt

# generate dummy data
N = 100
numSamps = 1000
data = np.random.random((N, 1, numSamps)).astype('float32')
labels = np.random.random((N, 1)).astype('float32')

val_data = np.random.random((N, 1, numSamps)).astype('float32')
val_labels = np.random.random((N, 1)).astype('float32')

X = np.linspace(-100, 100+1, numSamps)

for i in range(data.shape[0]):
    pure0 = np.sin(0.5*X)
    pure1 = np.sin(1*X)
    pure2 = np.sin(2*X)
    noise = np.random.normal(0, 1, numSamps)
    sig = np.zeros(X.shape)
    # pick 2 divider points
    a = np.random.randint(N/5, numSamps/2+1)
    b = np.random.randint(a+N/5, 2*numSamps/3+1)
    if i <= data.shape[0]/2:        
        sig[:a] = pure0[:a]
        sig[a:b] = pure1[a:b]
        sig[b:] = pure2[b:]
        label = 0
    else:
        sig[:a] = pure2[:a]
        sig[a:b] = pure1[a:b]
        sig[b:] = pure0[b:]      
        label = 1
    sig = sig + noise
    data[i,:,:] = sig
    labels[i] = label
# generat val data  
for i in range(val_data.shape[0]):
    pure0 = np.sin(0.5*X)
    pure1 = np.sin(1*X)
    pure2 = np.sin(5*X)
    noise = np.random.normal(0, 1, numSamps)
    sig = np.zeros(X.shape)
    # pick 2 divider points
    a = np.random.randint(0, numSamps/2)
    b = np.random.randint(a, numSamps+1)
    if i <= val_data.shape[0]/2:        
        sig[:a] = pure0[:a]
        sig[a:b] = pure1[a:b]
        sig[b:] = pure2[b:]
        label = 0
    else:
        sig[:a] = pure2[:a]
        sig[a:b] = pure1[a:b]
        sig[b:] = pure0[b:]      
        label = 1
    sig = sig + noise
    val_data[i,:,:] = sig
    val_labels[i] = label
    
# plt.figure(figsize=(9,3))
# plt.plot(X, np.squeeze(data[N/2,:,:]), 'k')
# plt.axis([-100, 100, -5, 5])
# plt.xlabel('time')
# plt.savefig('/home/haidar/Dropbox/Research/WaveletDeconv Poster/pos.png')
# 
# plt.figure(figsize=(9,3))
# plt.plot(X, np.squeeze(data[N/2+1,:,:]), 'r')
# plt.axis([-100, 100, -5, 5])
# plt.xlabel('time')
# plt.savefig('/home/haidar/Dropbox/Research/WaveletDeconv Poster/neg.png')
# plt.show()

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from WaveletDeconvolution import WaveletDeconvolution

inp_shape = data.shape[1:]
modelWD = Sequential()
modelWD.add(WaveletDeconvolution(5, filter_length=500, input_shape=inp_shape, padding='same'))
modelWD.add(Activation('tanh'))
modelWD.add(Convolution2D(5, (3, 3), padding='same'))
modelWD.add(Activation('relu'))
#end convolutional layers
modelWD.add(Flatten())
modelWD.add(Dense(25))
modelWD.add(Activation('relu'))
modelWD.add(Dense(1))
modelWD.add(Activation('sigmoid'))
modelWD.compile(optimizer='sgd', loss='binary_crossentropy')

print('Testing saving...')
modelWD.save('testWD_model.h5', overwrite=True)
print("Success!")

print('Testing loading...')
from keras.models import load_model
model = load_model('testWD_model.h5', custom_objects={'WaveletDeconvolution': WaveletDeconvolution})
print("Success!")

num_epochs = 20
plt.figure(figsize=(6,6))
Widths = np.zeros((num_epochs, 5)).astype('float32')
for i in range(num_epochs):
    hWD = model.fit(data, labels, epochs=1, batch_size=2, validation_data=(val_data, val_labels), verbose=0)
    print('Epoch %3d | train_loss: %.4f | val_loss: %.4f' % (i+1, hWD.history['loss'][0], hWD.history['val_loss'][0]))
    Widths[i,:] = model.layers[0].weights[0].eval()
    plt.plot(i, hWD.history['loss'][0], 'k.')
    plt.plot(i, hWD.history['val_loss'][0], 'r.')

plt.figure(figsize=(6,6))
for i in range(Widths.shape[1]):
   plt.plot(range(num_epochs), Widths[:,i]) 



plt.show()
