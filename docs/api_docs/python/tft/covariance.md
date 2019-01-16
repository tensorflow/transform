<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.covariance" />
<meta itemprop="path" content="Stable" />
</div>

# tft.covariance

``` python
tft.covariance(
    x,
    dtype,
    name=None
)
```

Computes the covariance matrix over the whole dataset.

The covariance matrix M is defined as follows:
Let x[:j] be a tensor of the jth element of all input vectors in x, and let
u_j = mean(x[:j]). The entry M[i,j] = E[(x[:i] - u_i)(x[:j] - u_j)].
Notice that the diagonal entries correspond to variances of individual
elements in the vector, i.e. M[i,i] corresponds to the variance of x[:i].

#### Args:

* <b>`x`</b>: A rank-2 `Tensor`, 0th dim are rows, 1st dim are indices in each input
    vector.
* <b>`dtype`</b>: Tensorflow dtype of entries in the returned matrix.
* <b>`name`</b>: (Optional) A name for this operation.


#### Raises:

* <b>`ValueError`</b>: if input is not a rank-2 Tensor.


#### Returns:

A rank-2 (matrix) covariance `Tensor`