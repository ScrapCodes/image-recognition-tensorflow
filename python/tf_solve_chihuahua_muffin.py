#@title MIT License
#
# Copyright (c) 2018 IBM
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import glob, os
import re

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image

# Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
def jpeg_to_8_bit_greyscale(path, maxsize):
	img = Image.open(path).convert('L')   # convert image to 8-bit grayscale
	# Make aspect ratio as 1:1, by applying image crop.
    # Please note, croping works for this dataset, but in general one 
    # needs to locate the subject and then crop or scale accordingly.
	WIDTH, HEIGHT = img.size
	if WIDTH != HEIGHT:
		m_min_d = min(WIDTH, HEIGHT)
		img = img.crop((0, 0, m_min_d, m_min_d))
	# Scale the image to the requested maxsize by Anti-alias sampling.
	img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
	return np.asarray(img) # convert shape (X, X) into (1, X, X)

class_names = ['chihuahua', 'muffin']

def load_image_dataset(path_dir, maxsize):
	images = []
	labels = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		img = jpeg_to_8_bit_greyscale(file, maxsize)
		if re.match('chihuahua.*', file):
			images.append(img)
			labels.append(0)
		elif re.match('muffin.*', file):
			images.append(img)
			labels.append(1)
	return (np.asarray(images), np.asarray(labels))

def load_test_set(path_dir, maxsize):
	test_images = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		img = jpeg_to_8_bit_greyscale(file, maxsize)
		test_images.append(img)
	return (np.asarray(test_images))

maxsize = 75, 75

(train_images, train_labels) = load_image_dataset('/Users/prashant/deep-learning-datasets/chihuahua-muffin', maxsize)

(test_images, test_labels) = load_image_dataset('/Users/prashant/deep-learning-datasets/chihuahua-muffin/test_set', maxsize)
print(train_images.shape)

print(train_labels)


print(test_images.shape)
print(test_labels)

# scaling the images to values between 0 and 1.
train_images = train_images / 255.0

test_images = test_images / 255.0


# Setting up the layers.

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.04, nesterov=True)

model = keras.Sequential([
    keras.layers.Flatten(input_shape = maxsize),
  	keras.layers.Dense(128, activation=tf.nn.sigmoid),
  	keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


model.compile(optimizer=sgd, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=100)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions)
# Using matplotlib display images.
def display_images(images, labels, title = "Default"):
	plt.title(title)
	plt.figure(figsize=(10,10))
	grid_size = min(25, len(images))
	for i in range(grid_size):
		plt.subplot(5, 5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[labels[i]])


#display_images(test_images, np.argmax(predictions, axis = 1))
#plt.show() # unless we do a plt.show(), the image grid won't appear.

# Comparing different model size and how they perform against the challenge.

baseline_model = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = maxsize),
  		keras.layers.Dense(128, activation=tf.nn.sigmoid),
  		keras.layers.Dense(16, activation=tf.nn.sigmoid),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])


bigger_model = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = maxsize),
		keras.layers.Dense(1024, activation=tf.nn.sigmoid),
		keras.layers.Dropout(0.4),
  		keras.layers.Dense(512, activation=tf.nn.sigmoid),
		keras.layers.Dense(64, activation=tf.nn.sigmoid),
		keras.layers.Dense(8, activation=tf.nn.sigmoid),
    	keras.layers.Dense(2, activation=tf.nn.sigmoid)
	])


smaller_model = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = maxsize),
  		keras.layers.Dense(512, activation=tf.nn.relu),
		keras.layers.Dropout(0.25),
		keras.layers.Dense(256, activation=tf.nn.relu),
		keras.layers.Dropout(0.25),
  		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(32, activation=tf.nn.sigmoid),
    	keras.layers.Dense(2, activation=tf.nn.sigmoid)
	])

vgg_style_model = keras.models.Sequential([
#	keras.layers.LSTM(512),
	keras.layers.Conv1D(128, 3, activation='relu', input_shape = maxsize),
	keras.layers.Conv1D(128, 3, activation='relu'),
	keras.layers.MaxPooling1D(pool_size=2),
	keras.layers.Dropout(0.25),
	keras.layers.Conv1D(64, 3, activation='relu'),
	keras.layers.Conv1D(64, 3, activation='relu'),
	keras.layers.MaxPooling1D(pool_size=2),
	keras.layers.Dropout(0.25),
	keras.layers.Flatten(),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(2, activation='softmax')
	])
datagen = keras.preprocessing.image.ImageDataGenerator(
		featurewise_center=True,
        zoom_range=0.2, # randomly zoom into images
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
vgg_style_model.compile(loss='sparse_categorical_crossentropy',
 optimizer=keras.optimizers.Adam(lr=0.001),
 metrics=['accuracy','sparse_categorical_crossentropy'])

def plot_history(histories, key='sparse_categorical_crossentropy'):
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()
  plt.xlim([0, max(history.epoch)])

bigger_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy','sparse_categorical_crossentropy'])

baseline_model.compile(optimizer=sgd,
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy','sparse_categorical_crossentropy'])

smaller_model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy','sparse_categorical_crossentropy'])

# bigger_model_history = bigger_model.fit(train_images, train_labels,
# 	epochs=200,
# #	batch_size = 512,
# 	validation_data=(test_images, test_labels),
# 	verbose=2)

# smaller_model_history = smaller_model.fit(train_images, train_labels,
# 	epochs=200,
# 	validation_data=(test_images, test_labels),
# 	verbose=2)

# baseline_model_history = baseline_model.fit(train_images, train_labels,
# 	epochs=100,
# 	validation_data=(test_images, test_labels),
# 	verbose=2)
vgg_style_model_history = vgg_style_model.fit_generator(datagen.flow(train_images, train_labels),
	epochs=30,
	validation_data=(test_images, test_labels),
	verbose=2,
	workers=4)

plot_history([
              #('smaller', smaller_model_history),
              #('bigger', bigger_model_history),
			  ('vgg', vgg_style_model_history)])

#plot_history([('smaller', smaller_model_history)])

predictions = smaller_model.predict(test_images)
predictions2 = bigger_model.predict(test_images)

predictions3 = vgg_style_model.predict(test_images)
display_images(test_images, np.argmax(predictions3, axis = 1), title = "vgg")

#display_images(test_images, np.argmax(predictions2, axis = 1), title = "big")

#display_images(test_images, np.argmax(predictions, axis = 1), title = "small")
print(predictions)
print(predictions3)

plt.show()
