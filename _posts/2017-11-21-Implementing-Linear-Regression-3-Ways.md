---
layout: post
title:  "Implementing Linear Regression 3 Ways"
author: "Sarat"
---



Arthur Samuel described Machine Learning as: **"the field of study that gives computers the ability to learn without being explicitly programmed."** Any machine learning problem can be classified into one of the following types: **Supervised Learning,** **Unsupervised Learning**. In the case of supervised learning given an input **"X"**, you already know what the output should look like i.e you will be provided with the corresponding output labels **"y"**. On contrary in Unsupervised Learning you will be given only input **"X"** and you do not know how output should look like.

Further, Supervised Learning problem can be classified into **Regression** and **Classification**. If we are trying to predict continous valued output the problem is considered as Regression. If we are trying to predict discrete valued output the problem is considered as Classification.

In this post I am going to implement an Univariate Linear Regression model in 3 ways: Scratch, Scikit-Learn, TensorFlow. I am calling it Univariate because the input has only one feature.

### Implementing from scratch


```python
# Importing Necessary Libraries
import numpy as np # Scientific Computing Library
import pandas as pd # Data Manipulation and Analysis Library
import matplotlib.pyplot as plt # Plotting Library

%matplotlib inline
```


```python
df = pd.read_csv("ex1data1.txt", header=None) # loading dataset
X_train = np.matrix(df[0].values).T # Converting dataframe into an numpy array
y = np.matrix(df[1].values).T 
```


```python
X = np.c_[np.ones(X_train.shape[0]), X_train] # Adding extra feature to every example
```


```python
m = X.shape[0]
np.random.seed(1)
W = np.random.randn(2,1)*0.01 # Weights
alpha = 0.001 # Learning Rate

for iter in range(500):
    z = np.dot(X,W)
    cost = (1/(2*m))*((z-y).T).dot(z - y)
    W = W - ((alpha/(m)*(X.T)).dot(z - y)) # Gradient Descent
Accuracy = np.mean(z-y)
print("cost: " + str(cost)) 
print("Train_Accuracy: "+str(Accuracy))
        
plt.scatter([X[:,1]],[y],s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.plot(X[:,1],z)


    
    

```

    cost: [[ 5.68901458]]
    Train_Accuracy: 0.657722472522
    

![](https://i.imgur.com/il1L3Px.png)


### Implementating using Scikit-Learn

Scikit-learn (formerly scikits.learn) is a machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = LinearRegression() #Estimator object
clf.fit(X_train,y_train) # Fitting data to estimator
predictions = clf.predict(X_test)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print("Train_Accuracy: " + str(train_accuracy))
print("Test_Accuracy: " + str(test_accuracy))
```

    Train_Accuracy: 0.751059106273
    Test_Accuracy: 0.521038287261
    


```python
# Visualization

plt.scatter([X_train[:,1]],[y_train],s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.plot(X_train[:,1],clf.predict(X_train))
```

![](https://i.imgur.com/7VtyeM0.png)


### Implementation using Tensor Flow

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks. It is used for both research and production at Google.


```python
import tensorflow as tf
```


```python
X = tf.placeholder("float64")
y = tf.placeholder("float64")

W = tf.Variable(np.random.randn(2,1)*0.01)
pred = tf.matmul(X, W)
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*m)
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        sess.run(optimizer, feed_dict={X:X_train,y:y_train})
    training_cost = sess.run(cost, feed_dict={X: X_train, y: y_train})
    print("Training cost=", training_cost)
    test_cost = sess.run(cost, feed_dict={X: X_test, y: y_test})
    print("Test cost=", test_cost)
    # Visualization
    plt.xlim(4,24)
    plt.plot(X_train, y_train, 'ro')
    plt.plot(X_train, X_train * sess.run(W))
    plt.legend()
    
```

    Training cost= 4.1585892567
    Test cost= 1.55391453545
    
![](https://i.imgur.com/DwQd5SF.png)

