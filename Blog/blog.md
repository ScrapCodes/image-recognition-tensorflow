# Solving a muffin or a Chihuahua, image recognition challenge with tensorflow and keras.

Deep neural network or Deep learning has become very popular in last few years, thanks to all the breakthroughs in research. I wanted to use deep neural network for solving something other than hello world of image recognition i.e. MNIST handwritten letters recognition. After going through the first tutorial on tensorflow and keras library, this became the first image recognition challenge to solve. 
This solution applies the same techniques as given in, 

[https://www.tensorflow.org/tutorials/keras/basic_classification](https://www.tensorflow.org/tutorials/keras/basic_classification).

<img src="../chihuahua-muffin/full.jpg" alt="" width="300"/>

## Import the data.

### Git clone this repository.
```
$ git clone https://github.com/ScrapCodes/deep-learning-datasets.git
$ cd deep-learning-datasets
$ python 
>>>
```

### Import tensorflow, keras and other helper libraries. 

In this blog, we have used `tensorflow` and `keras` for running machine learning and `Pillow` python library for image processing.

Using pip these can be installed on Macos X as follows,

```
$ sudo pip install tensorflow matplotlib pillow
```

Importing the python libraries. 

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re

# Pillow
import PIL
from PIL import Image

```

### Load the data

A python function to preprocess input images. For images to be converted into numpy arrays, they must have same dimensions.

```python
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
	return np.asarray(img)
```

A python funciton to load the dataset from images, into numpy arrays.

```python
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
```

We should scale the images to some standard size smaller than actual image resolution. These images are about than 170x170 in size, so we scale them all to 100x100 for furthur processing.

```python
maxsize = 100, 100
```

To load the data, let's execute these functions and load training and test data sets as follows.

```python
(train_images, train_labels) = load_image_dataset('deep-learning-datasets/chihuahua-muffin', maxsize)

(test_images, test_labels) = load_image_dataset('deep-learning-datasets/chihuahua-muffin/test_set', maxsize)

```

* train_images and train lables is Training dataset
* test_images and test_labels is testing dataset for validating our model's performance against unseen data.


Last we define the class names for our dataset. Since this data has only two classes i.e. an image can either be a Chihuahua or a Muffin, we have `class_names` as follows.

```python
class_names = ['chihuahua', 'muffin']
```

## Explore the data

In this dataset we have only 16 training examples, 8 of each Chihuahua and Muffin images.

```python
train_images.shape
(16, 100, 100)
```

Each image has it's respective label - either a `0` or `1`. A `0`indicates a `class_names[0]` i.e. a `chihuahua` and `1` indicates `class_names[1]` i.e. a `muffin`.

```python
print(train_labels)
[0 1 1 0 1 0 0 1 1 0 0 1 1 0 0 1]
```

For test set, we have got only 6 examples, 3 for each class.

```python
test_images.shape
(6, 100, 100)
print(test_labels)
[0 0 0 1 1 1]
```
### Visualize the dataset.

Using `matplotlib.pyplot` python library, we can visualize our data. Make sure you have matplotlib library installed.

Following python helper function helps us draw these images on our screen.

```python
def display_images(images, labels):
	plt.figure(figsize=(10,10))
	grid_size = min(25, len(images))
	for i in range(grid_size):
		plt.subplot(5, 5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[labels[i]])
```

Let's visualize the training dataset, as follows :

```python
display_images(train_images, train_labels)
plt.show()
```
<insert image>
Note: the images are greyscale and cropped, as we preprocessed our images at the time of loading.

Similarly we can visualize our test data set. Our testing set is fairly limited, feel free to use google search and add more testing or even training examples and see how things improve or perform.

## Preprocess the data

### Scaling the images to values between 0 and 1.

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

## Build the model

### Setup the layers

We have used four layers in total, the first layer is to simply flatten the dataset into a single array and does not get training. Rest three layers are dense and use sigmoid as activation function. 

```python
# Setting up the layers.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
  	keras.layers.Dense(128, activation=tf.nn.sigmoid),
  	keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

```
### Compile the model
Optimizer is SGD, i.e. stochastic gradient descent. The details are alredy explained in tensorflow tutorial, so skipped in this blog.

```python
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

model.compile(optimizer=sgd, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

```

## Train the model


```python
model.fit(train_images, train_labels, epochs=100)
```

Last three training iterations appear as follows:

```
....
16/16 [==============================] - 0s 884us/step - loss: 0.3056 - acc: 1.0000
Epoch 98/100
16/16 [==============================] - 0s 966us/step - loss: 0.3025 - acc: 1.0000
Epoch 99/100
16/16 [==============================] - 0s 817us/step - loss: 0.2994 - acc: 1.0000
Epoch 100/100
16/16 [==============================] - 0s 923us/step - loss: 0.2963 - acc: 1.0000
<tensorflow.python.keras.callbacks.History object at 0x11f0f5250>

```
Please note: accuracy of 1.0 indicates overfit. We will look into this later.

## Evaluate accuracy
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
6/6 [==============================] - 0s 16ms/step
('Test accuracy:', 0.8333333134651184)

Test accuracy is less than training accuracy. This indicates model has overfit the data. There are techniques to overcome this problem, we will discuss those later. This model is a good example of the use of API, but very far from perfect.

With recent advances in image recognition and using more training data, we can perform much better on this dataset challenge.

## Make predictions

To make predictions we can simply call predict on the generated model.

```python
predictions = model.predict(test_images)
```

```
print(predictions)
[[0.730212   0.26978803]
 [0.5961833  0.40381673]
 [0.36453587 0.63546413]
 [0.44840658 0.5515934 ]
 [0.24870424 0.75129575]
 [0.343152   0.656848  ]]
```

Finally, lets see how our model evaluated on test set visually.

```python
display_images(test_images, np.argmax(predictions, axis = 1))
```
<insert image>
