# pca-compress

This experiment attempts to compress neural networks by rank-reducing their convolutional filters.

The basic idea is to find a low-dimensional basis that captures most of the "important" variation in the outputs of a linear layer. Then we can rank-reduce the linear layer by projecting it onto this basis. For convolutions, this means reducing the number of convolutional filters (and transforming them to a new basis), and then tacking on a trailing 1x1 convolution that brings the new basis back up to the old basis before hitting any non-linearities.

There are a few different ways to find the most "important" basis. The simplest way is to use PCA on the output channels, given a batch (or multiple batches) of inputs. This is primarily implemented in [pca_compress/stats.py](pca_compress/stats.py) and [pca_compress/project.py](pca_compress/project.py). Another way is to compute a Hessian matrix such that v'*H*v approximates the change in network outputs (e.g. the MSE or KL-divergence) that arises from projecting out the direction v. In this case, we can use eigendecomposition to find the basis which results in the least amount of change in the outputs. Hessian approximation is implemented in [pca_compress/hessapprox.py](pca_compress/hessapprox.py).

# Results

I saw very little evidence that this approach is better than regular, structured pruning. My hope was that re-training would not be necessary with a good enough projection basis, but that doesn't seem to be the case. On ImageNet with a ResNet-18, a 50% rank reduction results in a 10% drop in top-1 accuracy using my best pruning method (PCA) before retraining. After retraining, this error shrinks to under 2%, but it still doesn't beat other structured pruning results.

# Running the code

The files [train_imagenet.py](train_imagenet.py) and [train_mnist.py](train_mnist.py) run the ImageNet and MNIST experiments, respectively. The meat of the algorithm itself is inside the [pca_compress](pca_compress) directory.

Here's an example of running iterative pruning on a pre-trained ResNet-18 on ImageNet:

```
$ python train_imagenet.py --pretrained --batch 64 --evaluate --print-freq 2000 --tune-epochs 1 --prune-step 0.98 -a resnet18 /path/to/imagenet
```

Some of my experiments were fairly ad-hoc (for example, I ran some MNIST experiments by adding code to train_mnist.py to load a pre-trained model). I expect that this is fine, since at the end of the day researchers tend to dump the actual training code into a notebook and mess around with it in there anyway.
