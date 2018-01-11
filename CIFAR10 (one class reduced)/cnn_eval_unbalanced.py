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

labels = [5]					# Reduce the 'dog' class


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
print ("Original CIFAR-10: ",x_train.shape, y_train.shape)

for label in labels:
	# Get number of samples of 'dog' and their indices
	indices = np.where(y_train == label)[0]
	n = len(indices)
	print ("Original number of images with ",label, ": ",n)

	# Delete a random 60% of n
	delete_index = sample(list(indices), int(0.6*n))
	x_train = np.delete(x_train, delete_index, 0)
	y_train = np.delete(y_train, delete_index, 0)

	print ("New number of images with ",label, ": ",len(np.where(y_train == label)[0]))

print ("New CIFAR-10: ",x_train.shape, y_train.shape)


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
Results: (60% removed)

Original CIFAR-10:  (50000, 32, 32, 3) (50000, 1)
Original number of images with  5 :  5000
New number of images with  5 :  2000
New CIFAR-10:  (47000, 32, 32, 3) (47000, 1)
Train on 47000 samples, validate on 10000 samples

Time:  -425.95149874687195
Test loss: 1.13779500198
Test accuracy: 0.5998
             precision    recall  f1-score   support

          0       0.64      0.68      0.66      1000
          1       0.74      0.76      0.75      1000
          2       0.34      0.68      0.46      1000
          3       0.46      0.44      0.45      1000
          4       0.55      0.52      0.54      1000
          5       0.66      0.25      0.36      1000
          6       0.83      0.58      0.68      1000
          7       0.63      0.75      0.69      1000
          8       0.80      0.66      0.73      1000
          9       0.72      0.67      0.70      1000

avg / total       0.64      0.60      0.60     10000
'''

