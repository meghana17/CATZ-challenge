# Video prediction challenge by Weights and Biases 

# The goal
The goal is to predict the 6th frame given 5 consecutive previous frames.

# The dataset
The dataset is comprised of sequences extracted from GIFs of cats thanks to GIPHY! Each cat has its own directory, which contains a sequence of 6 images. There are 6421 sequences in the training set and 1475 in the test set. Each image is 96x96 pixels.

# Evaluation
[Perceptual distance](https://www.compuphase.com/cmetric.htm) metric is used on the validation set (lower values are better).
