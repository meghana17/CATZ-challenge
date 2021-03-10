# Video prediction challenge 

# The goal
The goal is to predict the 6th frame given 5 consecutive previous frames.

# The dataset
The dataset is comprised of sequences extracted from GIFs of cats thanks to GIPHY! Each cat has its own directory, which contains a sequence of 6 images. There are 6421 sequences in the training set and 1475 in the test set. Each image is 96x96 pixels.

![cat1](https://user-images.githubusercontent.com/14092419/110579273-74ce1700-818c-11eb-98f9-85e341c0adec.jpg)


# Evaluation
[Perceptual distance](https://www.compuphase.com/cmetric.htm) metric is used on the validation set (lower values are better).
