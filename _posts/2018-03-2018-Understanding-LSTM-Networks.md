---
layout: post
title: "Understanding LSTM Networks"
author: "Sarat"
---

Traditional Neural Networks fails to generalize well for sequence data. Recurrent Neural Networks are a type of neural networks that have loop* within them and works well with sequence data. 
But the Recurrent Neural Networks works well for problems with short term dependencies. For example we want to predict the next word in the sentence "i drank apple ____" RNNs work well in this context but fail to generalize the words in a sentence that has long term dependencies like "the cat, which already ate...*was* full" and "the cats, which already ate...*were* full"
LSTM's works really good for long term dependencies. In Standard RNN's, the repeating module have very simple structure usually a single tanh layer whereas LSTM's have four neural network layers interacting in a very special way. The key to LSTM's is the memory cell state, horizontal line running on top of the diagram*.
LSTM's have ability to add or remove information to the cell state, carefully regulated by structures called gates.