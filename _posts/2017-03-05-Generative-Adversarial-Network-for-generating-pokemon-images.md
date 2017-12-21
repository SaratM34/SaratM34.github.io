---
layout: post
title: "Generative Adversarial Network For Generating Pokemon Images"
author: "Sarat"
---


# Generative Adversarial Networks! 

## Demo - We will generate new Pokemon using a Generative Adversarial Network!
![alt text](https://i.imgur.com/3wW3WWM.jpg "Logo Title Text 1")

## What is the difference between a Generative and a discriminative model?

![alt text](https://image.slidesharecdn.com/spatiallycoherentlatenttopicmodelforconcurrentobjectv1-3-091108054619-phpapp01/95/spatially-coherent-latent-topic-model-for-concurrent-object-segmentation-and-classification-5-728.jpg?cb=1257665696 "Logo Title Text 1")

![alt text](https://i.imgur.com/6Jye8hd.png "Logo Title Text 1")

### Generative Models:

- Aim to model how the data is generated. From P(x|c)×P(c)P(x|c)×P(c) we can obtain P(c|x)P(c|x) (see Bayes' Theorem). - They try to learn a joint probability distribution P(x,c)P(x,c) 

#### Pros:

- We have the knowledge about the data distribution.

#### Cons:

- Very expensive to get (a lot of parameters)
- Need lots of data

### Discriminative Models:

- Aim at learning P(c|x)P(c|x) by using probabilistic approaches (e.g., logistic regression), or by mapping classes from a set of points (e.g., perceptrons and SVMs). 

#### Pros:

- Easy to model

#### Cons:

- To classify, but not to generate the data.

![alt text](https://i.imgur.com/MGGLDsi.png "Logo Title Text 1")

![alt text](https://image.slidesharecdn.com/ch20-161213234552/95/deep-generative-models-2-638.jpg?cb=1481672832 "Logo Title Text 1")

#### So discriminative algorithms learn the boundaries between classeswhile generative algorithms learn the distribution of classes 

## What is a Generative Adversarial Network? 

![alt text](https://d3ansictanv2wj.cloudfront.net/GAN_Overall-7319eab235d83fe971fb769f62cbb15d.png "Logo Title Text 1")

![alt text](https://cdn-images-1.medium.com/max/958/1*-gFsbymY9oJUQJ-A3GTfeg.png "Logo Title Text 1")

#### Generator 
- Draw some parameter z from a source of randomness, e.g. a normal distribution
- Apply a function f such that we get x′=G(u,w)x′=G(u,w)
- Compute the gradient with respect to ww to minimize logp(y=fake|x′)log⁡p(y=fake|x′)

#### Discriminator 
- Improve the accuracy of a binary classifier f, i.e. maximize logp(y=fake|x′)log⁡p(y=fake|x′) and logp(y=true|x)log⁡p(y=true|x) for fake and real data respectively.

- There are two optimization problems running simultaneously, and the optimization terminates if a stalemate has been reached. 

- The models play two distinct (literally, adversarial) roles. 
- Given some real data set R
- G is the generator, trying to create fake data that looks just like the genuine data
- D is the discriminator, getting data from either the real set or G and labeling the difference. 
- G was like a team of forgers trying to match real paintings with their output, while D was the team of detectives trying to tell the difference. 
- Both D and G get better over time until G had essentially becomes a “master forger” of the genuine article and D is at a loss, “unable to differentiate between the two distributions.”

![alt text](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2017/04/image28.png "Logo Title Text 1")

#### There are really only 5 components to think about:

- R: The original, genuine data set
- I: The random noise that goes into the generator as a source of entropy
- G: The generator which tries to copy/mimic the original data set
- D: The discriminator which tries to tell apart G’s output from R

##### The actual ‘training’ loop is where we teach G to trick D and D to beware G.

## Use cases

![alt text](https://phillipi.github.io/pix2pix/images/teaser_v3.png "Logo Title Text 1")

![alt text](https://qph.ec.quoracdn.net/main-qimg-b85f35dcdcb5f4f48e8063dbf1f6abd3.webp "Logo Title Text 1")

- Generate images/videos/text/any data type!

Researchers from Insilico Medicine proposed an approach of artificially intelligent drug discovery using GANs.
The goal is to train the Generator to sample drug candidates for a given disease as precisely as possible to existing drugs from a Drug Database.

![alt text](https://cdn-images-1.medium.com/max/1600/0*--g8RQpR-Hofpa8G. "Logo Title Text 1")

After training, it’s possible to generate a drug for a previously incurable disease using the Generator, and using the Discriminator to determine whether the sampled drug actually cures the given disease.

## Other Types of GANs 

### Deep Convolutional GANs (DCGANs)

#### DCGANs were the first major improvement on the GAN architecture. They are more stable in terms of training and generate higher quality samples.

![alt text](https://image.slidesharecdn.com/dcgan-howdoesitwork-160923005917/95/dcgan-how-does-it-work-12-638.jpg?cb=1493068156 "Logo Title Text 1")

Important Discoveries

- Batch normalization is a must in both networks.
- Fully hidden connected layers are not a good idea.
- Avoid pooling, simply stride your convolutions!
- ReLU activations are your friend (almost always).
- Vanilla GANs could work on simple datasets, but DCGANs are far better.
- DCGANS are solid baseline to compare with your fancy new state-of-the-art GAN algorithm.

### Conditional GANs

![alt text](http://guimperarnau.com/files/blog/Fantastic-GANs-and-where-to-find-them/cGAN_overview.jpg "Logo Title Text 1")

#### CGANs use extra label information. This results in better quality images and being able to control – to an extent – how generated images will look.


- Here we have conditional information Y that describes some aspect of the data.
- if we are dealing with faces, Y could describe attributes such as hair color or gender. 
- Then, this attribute information is inserted in both the generator and the discriminator.

![alt text](http://guimperarnau.com/files/blog/Fantastic-GANs-and-where-to-find-them/cGAN_disentanglement.jpg "Logo Title Text 1")

### Conditional GANs are interesting for two reasons:

1. As you are feeding more information into the model, the GAN learns to exploit it and, therefore, is able to generate better samples.
2. We have two ways of controlling the representations of the images. Without the conditional GAN, all the image information was encoded in Z. With cGANs, as we add conditional information Y, now these two — Z and Y — will encode different information. For example, let’s suppose Y encodes the digit of a hand-written number (from 0 to 9). Then, Z would encode all the other variations that are not encoded in Y. That could be, for example, the style of the number (size, weight, rotation, etc).

### Wasserstein GANs

#### WGANs Change the loss function to include the Wasserstein distance. As a result, WassGANs have loss functions that correlate with image quality. Also, training stability improves and is not as dependent on the architecture.

![alt text](http://guimperarnau.com/files/blog/Fantastic-GANs-and-where-to-find-them/crazy_loss_function.jpg "Logo Title Text 1")
WTFFFFfff how about this instead...
![alt text](http://guimperarnau.com/files/blog/Fantastic-GANs-and-where-to-find-them/WassGAN_loss_function.jpg "Logo Title Text 1")

- GANs have always had problems with convergence and, as a consequence, you don’t really know when to stop training them. In other words, the loss function doesn’t correlate with image quality. This is a big headache because:

1. you need to be constantly looking at the samples to tell whether you model is training correctly or not.
2. you don’t know when to stop training (no convergence).
3. you don’t have a numerical value that tells you how well are you tuning the parameters.
4. For example, see these two uninformative loss functions plots of a DCGAN perfectly able to generate MNIST samples:

This interpretability issue is one of the problems that Wasserstein GANs aims to solve. How? GANs can be interpreted to minimize the Jensen-Shannon divergence, which is 0 if the real and fake distribution don’t overlap (which is usually the case). So, instead of minimizing the JS divergence, the authors use the Wasserstein distance, which describes the distance between the “points” from one distribution to the other.

So, WassGAN has a loss function that correlates with image quality and enables convergence. It is also more stable, meaning that it is not as dependent on the architecture. For example, it works quite well even if you remove batch normalization or try weird architectures.

### use if 
- you are looking for a state-of-the-art GAN with the highest training stability.
- you want an informative and interpretable loss function.




