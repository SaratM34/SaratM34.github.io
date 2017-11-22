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
df = pd.read_csv("ex1data1.txt", header=None)
X_train = np.matrix(df[0].values).T
y = np.matrix(df[1].values).T
```


```python
X = np.c_[np.ones(X_train.shape[0]), X_train]
```


```python
m = X.shape[0]
np.random.seed(1)
W = np.random.randn(2,1)*0.01
alpha = 0.001
for iter in range(500):
    z = np.dot(X,W)
    cost = (1/(2*m))*((z-y).T).dot(z - y)
    W = W - ((alpha/(m)*(X.T)).dot(z - y))
    if(iter%100==0):
        print("cost: " + str(cost))
        
plt.scatter([X[:,1]],[y],s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');
plt.plot(X,z)
plt.plot(X,z)

    
    

```

    cost: [[ 32.37838392]]
    cost: [[ 5.87654595]]
    cost: [[ 5.82698203]]
    cost: [[ 5.77917421]]
    cost: [[ 5.7330594]]
    




    [<matplotlib.lines.Line2D at 0x1f2a9f185c0>,
     <matplotlib.lines.Line2D at 0x1f2aa172ef0>]




![png](numpy_files/numpy_5_2.png)



```python

```
