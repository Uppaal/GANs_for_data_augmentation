import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time
import numpy as np
from sklearn.metrics import classification_report

# Load data: shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define parameters
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28	# Input image dimensions

label = 0


'''Preprocess'''

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

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

Time:  -390.4058084487915
Test loss: 0.287974014056
Test accuracy: 0.8958
             precision    recall  f1-score   support

          0       0.81      0.88      0.84      1000
          1       0.98      0.98      0.98      1000
          2       0.81      0.85      0.83      1000
          3       0.91      0.88      0.90      1000
          4       0.81      0.87      0.84      1000
          5       0.98      0.97      0.97      1000
          6       0.76      0.62      0.69      1000
          7       0.95      0.96      0.95      1000
          8       0.97      0.98      0.98      1000
          9       0.96      0.97      0.96      1000

avg / total       0.90      0.90      0.89     10000
'''
