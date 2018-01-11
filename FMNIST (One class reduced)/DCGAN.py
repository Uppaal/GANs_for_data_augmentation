# Reference: https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py

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

K.set_image_dim_ordering('th')

''' Preprocessing '''

# The results are a little better when the dimensionality of the random vector is only 10. The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Load FMNIST data
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
input_shape = (1, 28, 28)

# Get number of samples of 'T-Shirt' category (label 0) and their indices
label = 0
indices = np.where(y_train_full == label)[0]	
class_size = len(indices)
X_train = X_train_full[indices]
y_train = y_train_full[indices]

# Delete a random 70% of the class
delete_index = sample(range(len(X_train)), int(0.7 * class_size))
X_train = np.delete(X_train_full[indices], delete_index, 0)
y_train = np.delete(y_train_full[indices], delete_index, 0)

# Normalize data
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]



''' Model Definition '''

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

''' Generator '''
generator = Sequential()

generator.add(Dense(256*7*7, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((256, 7, 7)))

generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(0.2))

generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(0.2))

generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)


''' Discriminator '''
discriminator = Sequential()

discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=input_shape, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)


''' Combined network '''
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []



# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images_loss/dcgan_loss_epoch_%d.png' % epoch)


# Create a dataset of final generated images
def create_dataset(examples, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    print ("Generate: ", generatedImages.shape, type(generatedImages))

    # Save images to file
    np.save(('images/x_train_class_%d.npy' % label), generatedImages)
    print ("Saved X")

    y = np.zeros(generatedImages.shape[0])
    y[:] = label
    np.save(('images/y_train_class_%d.npy' % label), y)
    print ("Saved Y")

    # Plot some images
    plt.clf()
    for i in range(10):   # To create all required images, range(len(generatedImages)+1))
        plt.imshow(generatedImages[i, 0], cmap='gray_r')
        plt.axis('off')
        plt.savefig('images/%d.png' % i)


# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images_epoch/dcgan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print ('Epochs:', epochs)
    print ('Batch size:', batchSize)
    print ('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batchCount))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 5 == 0:
            plotGeneratedImages(e)
            saveModels(e)
            plotLoss(e)	# Plot losses from every epoch

if __name__ == '__main__':
    print ("\nDataset: ", len(X_train_full))
    print ("Class Size: ", class_size)
    print ("Reduced Class Size: ", len(X_train))
    print ("Generate", (class_size - len(y_train)), "samples\n")
    train(200, 64)
    create_dataset(examples=(class_size - len(y_train)))


