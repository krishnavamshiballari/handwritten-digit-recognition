from emnist import extract_training_samples
from emnist import extract_test_samples
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np
import tensorflow as tf

train_x, train_y = extract_training_samples('byclass')
test_x, test_y = extract_test_samples('byclass')



train_x = train_x.reshape((train_x.shape[0], 28, 28, 1)).astype('float32')
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1)).astype('float32')

train_x = train_x / 255
test_x = test_x / 255


train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)

num_classes = test_y.shape[1]

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
opti = tf.keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy', optimizer= opti , metrics=['accuracy'])

model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=50, batch_size=200)


scores = model.evaluate(test_x, test_y, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
model.save('emnist_.h5')
