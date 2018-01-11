import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from random import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint

K.set_image_dim_ordering('th')


''' Preprocessing '''

randomDim = 100

# Load FMNIST data
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
input_shape = (1, 28, 28)


X_train, y_train = X_train_full, y_train_full
X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1], input_shape[2])

# Normalize data
X_train = (X_train.astype(np.float32) - 127.5)/127.5


''' Model (Discriminator) Definition '''

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

discriminator = Sequential()

discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=input_shape, kernel_initializer=initializers.RandomNormal(stddev=0.02), name='CONV1'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', name='CONV2'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='CONV3'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', name='CONV4'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid', name='FF1'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)


# Fit the model
discriminator.fit(X_train, y_train, validation_split=0.33, epochs=20, batch_size=128, verbose=1)
discriminator.save_weights('pretrain_discriminator_weights.h5')
