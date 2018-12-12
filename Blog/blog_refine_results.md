
# How to refine our machine learning model.

As part of my own learning, continuing from the [previous article](blog.md) improve our neural network model, by using some of the well known machine learning techniques. 

In the previous article, we have seen certain problems with our training. In this article we will address them and see if our results improve as we go.

## Problems observed in the previous solution.

### 1. Overfitting.
A model is considered to overfit, when it performs with great accuracy on the training data i.e. data that was used for training the model, but when evaluated against a test or unseen data set, it performs rather poorly. This happens because our model has overfit the data.

Training accuracy if higher than testing accuracy is a clear indicator of this phenomenon. Thankfully, there are some techniques available to solve this problem.

#### Model size.

First thing to look at the size of the model, i.e. number of units. If the model used is far more bigger than the problem at hand, it is more likely to learn the features/patterns not relevant to the problem and thus overfit to the training data. A larger model will not generalize well and a smaller model will underfit the data i.e. it can do better with more number of units.

Taking the model in our previous blog as a baseline, we will evaluate the result of reducing the size and increasing the size on the performance of the model.
<image src='model_size_comparision_plot.png' alt="" width="500"/>

#### Number of Epochs.

It is important to 

#### L1 and L2 Regularization.


#### Using Dropout.

Keras library provides a Droupout Layer, this concept was introduced by a paper _Dropout: A Simple Way to Prevent Neural Networks from Overfitting(JMLR 2014)_. Consequence of adding a dropout layer is the training time is increased and if the dropout is high then underfitting.

Can be solved by adding dropouts.
### 2. Lack of training data.
In a way with only 16 training examples, we have done reasonably well.
Can be solved by transfer learning.

#### Using Adaptive learning rate.

### 3. Convergence