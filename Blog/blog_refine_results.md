
# How to refine our deep learning model(Part 2 of [Solving a chihuahua or muffin](blog.md)).

As part of my own learning, continuing from the [previous article](blog.md) and trying to improve our neural network model, by using some of the well known machine learning techniques mentioned in [https://www.tensorflow.org/tutorials/keras/](https://www.tensorflow.org/tutorials/keras/). 

In the previous article, we have seen certain problems with our training. In this article we will address them and see if our results improve as we go.

## Problems observed in the previous solution.

### 1. Overfitting.
A model is considered to overfit, when it performs with great accuracy on the training data i.e. data that was used for training the model, but when evaluated against a test or unseen data set, it performs rather poorly. This happens because our model has overfit the data.

Training accuracy if higher than testing accuracy is a clear indicator of this phenomenon. Thankfully, there are some techniques available to solve this problem.

#### Model size.

First thing to look at the size of the model, i.e. number of units. If the model used is far more bigger than the problem at hand, it is more likely to learn the features/patterns not relevant to the problem and thus overfit to the training data. A larger model will not generalize well and a smaller model will underfit the data i.e. it can do better with bigger model.

Taking the model in our previous blog as a baseline, we will evaluate the result of reducing the size and increasing the size on the performance of the model.

Following models were tried and compared.

```python

baseline_model = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
  		keras.layers.Dense(128, activation=tf.nn.sigmoid),
  		keras.layers.Dense(16, activation=tf.nn.sigmoid),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])

bigger_model2 = keras.models.Sequential([
		keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
		keras.layers.Dense(1024, activation=tf.nn.relu),
		keras.layers.Dense(512, activation=tf.nn.relu),
		keras.layers.Dense(64, activation=tf.nn.relu),
		keras.layers.Dense(16, activation=tf.nn.relu),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])

bigger_model1 = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
  		keras.layers.Dense(512, activation=tf.nn.relu),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(16, activation=tf.nn.relu),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])

smaller_model1 = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
 		keras.layers.Dense(64, activation=tf.nn.relu),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])

```
To determine the ideal model, we plot loss function of validation data against number of epochs.

1. Comparison of Smaller, bigger and baseline models.
<image src='model_size_comparision_plot2.png' alt="" width="500"/>

2. Comparison of bigger, bigger2 and baseline models.

<image src='model_size_comparision_plot3.png' alt="" width="500"/>

In these plots it is observed, validation loss i.e. `sparse_categorical_crossentropy`, is almost similar for `bigger` and `bigger2` models, and better than `smaller` and `baseline` models. So we go ahead and select these models over our `baseline` model for further tuning.

#### Number of Epochs.

Number of epochs plays an important role in avoiding overfitting and overall model performance, in the comparison graphs plotted in above section, we observe the loss function for validation data reaches a minimum and then on further training increases again, while loss function of training data reduces further. This is exactly what overfitting means, model learns patterns that are very specific to dataset and does not generalize well. So it does better with training data than validation data. We have to stop, before the model overfits the data. So in above case `epoch` value of `40` would be ideal.

#### L1 and L2 Regularization.

The effect of applying L2 regularization is that of adding some random noise to the layers. Plot below shows the effect of applying this on our model.

<image src='L2_regularization.png' alt="" width="600"/>

#### Using Dropout.

Keras library provides a Dropout Layer, this concept was introduced by a paper _Dropout: A Simple Way to Prevent Neural Networks from Overfitting(JMLR 2014)_. Some negative consequences of adding a dropout layer is that, the training time is increased and if the dropout is high then underfitting.

Models after applying the Dropout layers.

```python
bigger_model1 = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
  		keras.layers.Dense(512, activation=tf.nn.relu),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(16, activation=tf.nn.relu),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])

bigger_model2 = keras.models.Sequential([
		keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
		keras.layers.Dense(1024, activation=tf.nn.relu),
		keras.layers.Dropout(0.5),
  		keras.layers.Dense(512, activation=tf.nn.relu),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(64, activation=tf.nn.relu),
		keras.layers.Dense(16, activation=tf.nn.relu),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])
```

Effect of applying dropout regularization,

<image src='dropout_regularization.png' alt="" width="600"/>

During one of the run, the bigger model did not converge at all, even after 250 epochs. This is one of the side effects of applying dropout regularization.

<image src='Model_No_converge.png' alt="" width="600"/>

### 2. Lack of training data.
In a way with only 26 or so training examples, we have done reasonably well. But, for image processing there are several techniques of data augmentation by applying some distortion to original image and generating more data. For example, for every input image we can have a invert color image added to our dataset. So, to achieve this, the `load_image_dataset` function(_from previous blog article_) is modified as follows. It is also possible to add a randomly rotated image for each original image.

```python
# invert_image if true, also stores an invert color version of each image in the training set.
def load_image_dataset(path_dir, maxsize, reshape_size, invert_image=False):
	images = []
	labels = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		img = jpeg_to_8_bit_greyscale(file, maxsize)
		inv_image = 255 - img # Generate a invert color image of the original.

		if re.match('chihuahua.*', file):
			images.append(img.reshape(reshape_size))
			labels.append(0)
			if invert_image:
				labels.append(0)
				images.append(inv_image.reshape(reshape_size))
		elif re.match('muffin.*', file):
			images.append(img.reshape(reshape_size))
			labels.append(1)
			if invert_image:
				images.append(inv_image.reshape(reshape_size))
				labels.append(1)
	return (np.asarray(images), np.asarray(labels))
```

Effects of adding invert color images and randomly rotating images, on training with dropout on, is as follows.
The size of dataset increased by 3X.
<image src='sigmoid_dropout_25.png' alt="" width="600"/>

The result indicate, this has worsened the overfit of the data.

_Please note: For data augmentation, keras provides a inbuilt utility, `keras.preprocessing.image.ImageDataGenerator`, it is out of scope for the blog._

Another way to overcome the problem of less training data is to use a pretrained model and augment it with new training example. This approach is called transfer learning. Since tensorflow and keras provide a good mechanism for saving and loading models, this can be quite easily achieved. But out of scope for this blog.

## Conclusion
On further testing with different models and activation functions, the best results were observed by using sigmoid as activation function and dropout layer in our baseline model. A similar performance was observed with relu activation function, but with sigmoid, curve was smoother. Also the size of the image was reduced to 50x50, it improved the training time without impacting the performance of models.

Apart from the above, I have also tested a VGG style multilayer CNN model, and multiple variations of CNN models, but somehow the results were very poor with it. 

Plot of the results from all the three models,
<image src='conclusion.png' alt="" width="600"/>

Baseline model used.

```python
baseline_model = keras.models.Sequential([
    	keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
  		keras.layers.Dense(128, activation=tf.nn.sigmoid),
		keras.layers.Dropout(0.25),
  		keras.layers.Dense(16, activation=tf.nn.sigmoid),
    	keras.layers.Dense(2, activation=tf.nn.softmax)
	])

baseline_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy','sparse_categorical_crossentropy'])

```
Output:

```
- 0s - loss: 0.0217 - acc: 1.0000 - sparse_categorical_crossentropy: 0.0217 - val_loss: 0.2712 - val_acc: 0.9286 - val_sparse_categorical_crossentropy: 0.2712
Epoch 119/400
 - 0s - loss: 0.0224 - acc: 1.0000 - sparse_categorical_crossentropy: 0.0224 - val_loss: 0.2690 - val_acc: 0.9286 - val_sparse_categorical_crossentropy: 0.2690
Epoch 120/400
```

Results:

<image src='final_screenshot.png' alt="" width="600"/>

Next, I would like to improve my understandings of CNN and VGG style networks for image recognition and even more advanced usages of neural networks. 
