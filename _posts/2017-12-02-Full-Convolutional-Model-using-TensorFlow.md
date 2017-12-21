---
layout: post
title: "Full Convolutional Model using TensorFlow"
author: "Sarat"
---

## Introduction

In this post I implemented a full convolutional model using tensorflow that uses SIGNS dataset which contains hand signs of digits 0 to 5. Initliased weights with xaviers initialisation. The optimizer used is Adam Optimizer.

**Basic Architecture**
* Initialize parameters
* Forward Prop
* Compute Cost
* Back Prop done automatically by TensorFlow
* Predict
 

```python
import tensorflow as tf
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from cnn_utils import *

%matplotlib inline
np.random.seed(1)
```


```python
# Loading the data(signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
```


```python
index = 14
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 2
    


![](https://i.imgur.com/o8vDHE0.png)



```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
conv_layers = {}
```


```python
# Create Placeholders for input to feed during running session
def  create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X,Y
```


```python
# Initialising weights with xaviers initializer
def initialize_parameters():
    
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [4,4,3,8], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1":W1, "W2":W2}
    
    return parameters

```


```python
def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # Conv layer 1
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    
    # Conv layer 2
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME') 
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    
    # Fully-Connected layer without Non-Linear Activation
    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)
    
    return Z3
```


```python
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost
    
```


```python
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    X,Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer, cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                ### END CODE HERE ###
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters
            
            
```


```python
_, _, parameters = model(X_train, Y_train, X_test, Y_test)
```

    Cost after epoch 0: 1.917920
    Cost after epoch 5: 1.532475
    Cost after epoch 10: 1.014804
    Cost after epoch 15: 0.885137
    Cost after epoch 20: 0.766963
    Cost after epoch 25: 0.651208
    Cost after epoch 30: 0.613356
    Cost after epoch 35: 0.605931
    Cost after epoch 40: 0.534713
    Cost after epoch 45: 0.551402
    Cost after epoch 50: 0.496976
    Cost after epoch 55: 0.454438
    Cost after epoch 60: 0.455496
    Cost after epoch 65: 0.458359
    Cost after epoch 70: 0.450040
    Cost after epoch 75: 0.410687
    Cost after epoch 80: 0.469005
    Cost after epoch 85: 0.389253
    Cost after epoch 90: 0.363808
    Cost after epoch 95: 0.376132
    


![](https://i.imgur.com/fSWs8ep.png)


    Tensor("Mean_1:0", shape=(), dtype=float32)
    Train Accuracy: 0.868519
    Test Accuracy: 0.733333
    

## Conclusion

We only just need to perform forward propagation. Back Prop will be automatically carried out by the tensorflow. This a simple convolutional model that detects hand signs. The model achieved 86% train accuracy and 73% test accuracy. We can improve test set accuracy by tuning the hyperparameters.




