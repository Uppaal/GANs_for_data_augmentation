import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time
import numpy as np
from random import *
from sklearn.metrics import classification_report


# Load data
# x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define parameters
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols, img_depth = 32, 32, 3	# Input image dimensions



'''Preprocess'''

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], img_depth, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], img_depth, img_rows, img_cols)
	input_shape = (img_depth, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth)
	input_shape = (img_rows, img_cols, img_depth)


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




'''
Results:


'''

