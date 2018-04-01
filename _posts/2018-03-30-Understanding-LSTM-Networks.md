---
layout: post
title: "Understanding LSTM Networks"
author: "Sarat"
---

Traditional Neural Networks fails to generalize well for sequence data. Recurrent Neural Networks are a type of neural networks that have loop within them and works well with sequence data.
<div align="center">
<img src="https://i.imgur.com/xH3VMry.png" alt="An unrolled RNN">
<strong>An unrolled RNN structure</strong>
</div><br />

But the Recurrent Neural Networks works well for problems with short term dependencies. For example we want to predict the next word in the sentence "i drank apple ____" RNNs work well in this context but fail to generalize the words in a sentence that has long term dependencies like "the cat, which already ate...*was* full" and "the cats, which already ate...*were* full"
LSTM's works really good for long term dependencies. In Standard RNN's, the repeating module have very simple structure usually a single "tanh" layer whereas LSTM's have four neural network layers interacting in a very special way. The key to LSTM's is the memory cell state, horizontal line running on top of the diagram.

<div align="center">
<img src="https://i.imgur.com/waWHYQE.png" alt="">
<strong>Single LSTM Unit</strong>
</div><br />

LSTM's have ability to add or remove information to the cell state, carefully regulated by structures called gates.

<div align="center">
<img src="https://i.imgur.com/y9XGWsj.png" alt="">
<strong>Forget Gate</strong>
</div><br />

The first sigmoid layer outputs numbers between 0 and 1, describing how much of each component should be let through. "0" means "let nothing through" and "1" mean "let everything through".
for example, if we would like to forget the gender of the old subject and update to new gender of another subject. The first sigmoid layer looks at "previous cell activation" and "input at the current time step" and outputs a value between 0 and 1.

<div align="center">
<img src="https://i.imgur.com/Ooxr3Ql.png" alt="">
<strong>Update Gate</strong>
</div><br />

<div align="center">
<img src="https://i.imgur.com/59Br1vk.png" alt="">
<strong>Updating candidate values</strong>
</div><br />


 In order to update it involves two steps: first a sigmoid layer called input gate layer decides which values we will update and next "tanh" layer creates vector of candidate values that could be added to state. In next step, we will combine these two and create and update to the state.

 <div align="center">
 <img src="https://i.imgur.com/XAdGY9y.png" alt="">
 <strong>Output Gate</strong>
 </div><br />

 Next, we need to output. The output will be based on our cell state but filtered version. First we run a sigmoid layer that decides which parts of the cell state we are going to output. Then, we put the cell state through 'tanh' to push the cell state values between (-1,1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

 **Variation of LSTM**

 <div align="center">
 <img src="https://i.imgur.com/ZLM5vBa.png" alt="">
 <strong>Output Gate</strong>
 </div><br />

 A simpler and powerful variation of LSTM is called GRU(Gated Recurrent Unit). It combines the forget gate and input gates into single "update gate". It also merges cell state and hidden state, and make some other changes.

 **References:**
 * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
