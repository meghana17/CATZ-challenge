# Video prediction challenge 

## The goal
The goal is to predict the 6th frame given 5 consecutive previous frames.

## The dataset
The dataset is comprised of sequences extracted from GIFs of cats thanks to GIPHY! Each cat has its own directory, which contains a sequence of 6 images. There are 6421 sequences in the training set and 1475 in the test set. Each image is 96x96 pixels.

![cat1](https://user-images.githubusercontent.com/14092419/110579273-74ce1700-818c-11eb-98f9-85e341c0adec.jpg)


## Evaluation
[Perceptual distance](https://www.compuphase.com/cmetric.htm) metric is used on the validation set (lower values are better).

## Work done
1. Image augmentation

2. Architecture 1 - CNN and LSTM as propsed in [1] and [2]
   Used Dropout and Gaussian noise to prevent overfitting on the training data
   Created five different models and used bagging to reduce the perceptual distance
   
3. Architecture 2 - Generative Adversarial Network to predict future frames of video as proposed in [3]


## References
[1] [Prednet](https://github.com/coxlab/prednet) proposed in [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104)
[2] [ConvLSTM](https://github.com/farquasar1/ConvLSTM) for video prediction
[3] [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440)
