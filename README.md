# pca-compress

This experiment attempts to compress a neural network initialization by applying dimensionality reduction to the activations and then projecting the weights into a lower-dimensional space that preserves the reduced-dimensionality activations.

I started with the PyTorch MNIST demo, and attempted to reduce the dimensionality of the weights. Reducing the number of filters in the first convolutional layer down to 8 (from 32) was surprisingly effective (compared to simply training a smaller network with 8 filters from the start). This reflects the fact that the activations from conv1 had only a few significant principal components. However, reducing deeper layers like conv2 and fc1 had a negative impact, as reflected by the heavy-tail distribution of their activations' principal components.
