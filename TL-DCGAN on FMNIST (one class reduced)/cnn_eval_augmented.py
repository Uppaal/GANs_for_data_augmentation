import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time
import numpy as np
from random import *
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define parameters
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols, img_depth = 28, 28, 1	# Input image dimensions

labels = [0]					# Reduce the 'top' class


'''Preprocess'''

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], img_depth, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], img_depth, img_rows, img_cols)
	input_shape = (img_depth, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth)
	input_shape = (img_rows, img_cols, img_depth)


# Unbalance data
print ("Original FMNIST: ",x_train.shape, y_train.shape)

for label in labels:
	# Get number of samples of 'top' and their indices
	indices = np.where(y_train == label)[0]
	n = len(indices)
	print ("Original number of images with ",label, ": ",n)

	# Delete a random 70% of n
	delete_index = sample(list(indices), int(0.7*n))
	x_train = np.delete(x_train, delete_index, 0)
	y_train = np.delete(y_train, delete_index, 0)

	print ("New number of images with ",label, ": ",len(np.where(y_train == label)[0]))
	print ("Now adding GAN data ...")

	# Load synthetic data
	x_gen = np.load('images/x_train_class_%d.npy' % label)

	# Normalize data
	x_gen = np.abs(x_gen)
	x_gen *= 255
	x_gen = x_gen.astype(np.uint8)
	x_gen = np.reshape(x_gen, (x_gen.shape[0], img_rows, img_cols, img_depth))
	
	y_gen = np.load('images/y_train_class_%d.npy' % label)
	y_gen = y_gen.astype(int)

	# Combine both datasets and shuffle
	x_train = np.concatenate((x_train, x_gen))
	y_train = np.concatenate((y_train, y_gen))
	x_train, y_train = shuffle(x_train, y_train, random_state=0)

print ("New FMNIST: ",x_train.shape, y_train.shape)


# Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



''' Train Model '''

# Define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# Fit model
start = time.time()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate model
end = time.time()
print ("Time: ", (start-end))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test)
print(classification_report(Y_test, y_pred))
