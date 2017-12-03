---
layout: post
title: "Implementing Logistic Regression 3 Ways"
author: "Sarat"
---

### Logistic Regression
In the previous post we have seen an Regression Algorithm called Linear Regression implemented in 3 ways. In this post I am going to talk about classification task and implement an algorithm called Logistic regression in 3 ways. Classification problems deal with discrete valued output. If there are two classes we call it binary classification problem and problems with more than two classes are called as multi class classification problems. In this post I am going to implement an binary classification problem that distinguishes Cats from Non-Cats.

**General Architecture for building an Algorithm**
* Initialize Parameters
* Calculate Cost and Gradients
* Update parameters using Optimization Algorithms(Gradient Descent)


```python
# Importing necessasry libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

%matplotlib inline
```


```python
# Loading train dataset
train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])
# Loading test dataset
test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])
train_Y = train_set_y_orig.reshape(1,train_set_y_orig.shape[0]) # Reshaping into size (1, num of examples)
test_Y = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])
classes = np.array(test_dataset["list_classes"][:])
```


```python
train_set_x_orig.shape
```




    (209, 64, 64, 3)




```python
train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#Preprocessing (Standardizing data)
train_X = train_X/255.
test_X = test_X/255.

m = train_X.shape[1]
num_feat = train_X.shape[0]
```


```python
# Initializing parameters

W = np.zeros((num_feat,1))
b = 0
learning_rate = 0.005

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
```


```python
# Forward Propagation, Back Propagation, Gradient Descent steps
costs = []
iterations = []
for i in range(2000):
    Z = np.dot(W.T,train_X)+b
    A = sigmoid(Z)
    cost = (-1/m)*np.sum((np.dot(train_Y, np.log(A).T)+np.dot((1-train_Y),np.log(1-A).T)))
    dw = (1/m) * np.dot(train_X, (A-train_Y).T)
    db = (1/m) * np.sum((A-train_Y))
    W = W - learning_rate * dw
    b = b - learning_rate * db
    if(i%100==0):
        iterations.append(i)
        costs.append(cost)
        print(cost)
parameters = {"W":W,"b":b}

```

    0.0528243277002
    0.0520045086156
    0.0512088639306
    0.0504363575963
    0.0496860114665
    0.0489569013262
    0.0482481532411
    0.0475589401991
    0.046888479016
    0.0462360274824
    0.0456008817297
    0.0449823737963
    0.044379869376
    0.0437927657338
    0.0432204897734
    0.0426624962455
    0.0421182660831
    0.041587304856
    0.0410691413315
    0.0405633261357
    


```python
# Prediction Function
def pred(X):
    pred = sigmoid(np.dot(W.T,X)+b)
    for i in range(X.shape[1]):
        if pred[0,i]>=0.5:
            pred[0,i]=1
        else:
            pred[0,i]=0
    return pred
print("train accuracy: {} %".format(100 - np.mean(np.abs(pred(train_X)-train_Y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(pred(test_X)-test_Y)) * 100))
```

    train accuracy: 100.0 %
    test accuracy: 68.0 %
    


```python
# Predicting using our Own Image
my_image = "download.jpg"

fname = "images/"+my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_predicted_image = pred(my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```

    C:\Users\user\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: RuntimeWarning: overflow encountered in exp
      
    

    y = 0.0, your algorithm predicts a "non-cat" picture.
    


![](https://i.imgur.com/0Af5fvc.png)



```python
# Plot learning curve (with costs)
#costs = np.squeeze(d['costs'])
plt.plot(iterations,costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.005))
plt.show()
```


![](https://i.imgur.com/MLdAWIJ.png)


### Implementation in Scikit-Learn

Sklearn has an inbuilt estimators to implement Logistic Regression. It is part of sklearn.linear_model package. Pickle is sklearn library used for model persistence i.e the model can be saved and later used directly without training the model everytime when we want to predict on new data.


```python
from sklearn.linear_model import LogisticRegression
import pickle
```


```python
clf = LogisticRegression()
clf.fit(train_X.T,train_Y.T) # Sklearn expects shape of the input to be in shape(n_samples, n_features)
train_accuracy = clf.score(train_X.T,train_Y.T)
test_accuracy = clf.score(test_X.T, test_Y.T)
print("Train_Accuracy: "+str(train_accuracy))
print("Test_Accuracy: "+str(test_accuracy))
predict = clf.predict(test_X.T)
print(predict)

s = pickle.dumps(clf)

```

    C:\Users\user\Anaconda3\lib\site-packages\sklearn\utils\validation.py:526: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    Train_Accuracy: 1.0
    Test_Accuracy: 0.72
    [1 1 1 1 1 0 0 1 1 1 0 0 1 1 0 1 0 1 0 0 1 0 0 1 0 1 1 0 0 1 0 1 1 1 0 0 0
     1 0 0 1 0 1 0 1 1 0 1 1 0]
    


```python
print(test_Y)
```

    [[1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 0 1 1 1 1 0 0 0
      1 0 0 1 1 1 0 0 0 1 1 1 0]]
	  
	  

### Implementation in TensorFlow


```python
import tensorflow as tf
import numpy as np
import h5py
```


```python
# Loading train dataset
train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])
train_set_y_orig = np.array(train_dataset["train_set_y"][:])
# Loading test dataset
test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])
test_set_y_orig = np.array(test_dataset["test_set_y"][:])
train_Y = train_set_y_orig.reshape(1,train_set_y_orig.shape[0]) # Reshaping into size (1, num of examples)
test_Y = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])
classes = np.array(test_dataset["list_classes"][:])
train_set_x_orig.shape
```




    (209, 64, 64, 3)




```python
train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_X = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#Preprocessing (Standardizing data)
train_X = train_X/255.
test_X = test_X/255.

X = tf.placeholder(tf.float32,[None,12288])
W = tf.Variable(tf.zeros([12288,1]))
b = 0


Y = tf.nn.sigmoid(tf.matmul(tf.reshape(X,[-1,12288]),W)+b)
Y_ = tf.placeholder(tf.float32, [None,1])

is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

cost = tf.reduce_mean(-tf.reduce_sum(Y_*tf.log(Y), reduction_indices=1))
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y, labels=Y_))
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

init = tf.global_variables_initializer()
```


```python
sess = tf.Session()
sess.run(init)
for i in range(2000):
    _, c = sess.run([optimizer, cost], feed_dict={X:train_X.T,Y_:train_Y.T})
    
    if i%500==0:
        print(c)

a,c = sess.run([accuracy,cost], feed_dict={X:train_X.T,Y_:train_Y.T})
te_a,te_c = sess.run([accuracy,cost], feed_dict={X:test_X.T,Y_:test_Y.T})
print("Train_accuracy: "+str(a))
print("Test_accuracy: "+str(te_a))


#pred = sess.run([Y],feed_dict={X:train_X.T[2:3]})
#print(pred)


            

```

    0.238788
    0.000950269
    0.000563892
    0.000407643
    Train_accuracy: 1.0
    Test_accuracy: 1.0
    

