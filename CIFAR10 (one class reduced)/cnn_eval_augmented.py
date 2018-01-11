import keras
from keras.datasets import cifar10
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
	y_gen = y_gen.reshape(y_gen.shape[0], 1)

	# Combine both datasets and shuffle
	x_train = np.concatenate((x_train, x_gen))
	y_train = np.concatenate((y_train, y_gen))
	x_train, y_train = shuffle(x_train, y_train, random_state=0)

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
Results: (60% reomved) (at 50 GAN epochs)
Time:  -435.34533071517944
Test loss: 1.1124170166
Test accuracy: 0.617
             precision    recall  f1-score   support

          0       0.59      0.74      0.66      1000
          1       0.79      0.71      0.75      1000
          2       0.44      0.54      0.49      1000
          3       0.42      0.54      0.48      1000
          4       0.56      0.58      0.57      1000
          5       0.71      0.19      0.30      1000
          6       0.69      0.77      0.73      1000
          7       0.69      0.68      0.69      1000
          8       0.71      0.76      0.73      1000
          9       0.75      0.66      0.70      1000

avg / total       0.64      0.62      0.61     10000

Results: (60% reomved) (at 115 GAN epochs)
Time:  -547.9734735488892
Test loss: 1.13643145638
Test accuracy: 0.6075
             precision    recall  f1-score   support

          0       0.66      0.67      0.66      1000
          1       0.75      0.74      0.74      1000
          2       0.42      0.53      0.47      1000
          3       0.46      0.47      0.46      1000
          4       0.52      0.58      0.55      1000
          5       0.68      0.22      0.33      1000
          6       0.58      0.82      0.67      1000
          7       0.70      0.68      0.69      1000
          8       0.72      0.74      0.73      1000
          9       0.73      0.64      0.68      1000

avg / total       0.62      0.61      0.60     10000

'''

