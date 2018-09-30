<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.pca" />
<meta itemprop="path" content="Stable" />
</div>

# tft.pca

``` python
tft.pca(
    x,
    output_dim,
    dtype,
    name=None
)
```

Computes pca on the dataset using biased covariance.

The pca analyzer computes output_dim orthonormal vectors that capture
directions/axes corresponding to the highest variances in the input vectors of
x. The output vectors are returned as a rank-2 tensor with shape
(input_dim, output_dim), where the 0th dimension are the components of each
output vector, and the 1st dimension are the output vectors representing
orthogonal directions in the input space, sorted in order of decreasing
variances.

The output rank-2 tensor (matrix) serves a useful transform purpose. Formally,
the matrix can be used downstream in the transform step by multiplying it to
the input tensor x. This transform reduces the dimension of input vectors to
output_dim in a way that retains the maximal variance.

NOTE: To properly use PCA, input vector components should be converted to
similar units of measurement such that the vectors represent a Euclidean
space. If no such conversion is available (e.g. one element represents time,
another element distance), the canonical approach is to first apply a
transformation to the input data to normalize numerical variances, i.e.
tft.scale_to_z_score(). Normalization allows PCA to choose output axes that
help decorrelate input axes.

Below are a couple intuitive examples of PCA.

Consider a simple 2-dimensional example:

Input x is a series of vectors [e, e] where e is Gaussian with mean 0,
variance 1. The two components are perfectly correlated, and the resulting
covariance matrix is
[[1 1],
 [1 1]].
Applying PCA with output_dim = 1 would discover the first principal component
[1 / sqrt(2), 1 / sqrt(2)]. When multipled to the original example, each
vector [e, e] would be mapped to a scalar sqrt(2) * e. The second principal
component would be [-1 / sqrt(2), 1 / sqrt(2)] and would map [e, e] to 0,
which indicates that the second component captures no variance at all. This
agrees with our intuition since we know that the two axes in the input are
perfectly correlated and can be fully explained by a single scalar e.

Consider a 3-dimensional example:

Input x is a series of vectors [a, a, b], where a is a zero-mean, unit
variance Gaussian. b is a zero-mean, variance 4 Gaussian and is independent of
a. The first principal component of the unnormalized vector would be [0, 0, 1]
since b has a much larger variance than any linear combination of the first
two components. This would map [a, a, b] onto b, asserting that the axis with
highest energy is the third component. While this may be the desired
output if a and b correspond to the same units, it is not statistically
desireable when the units are irreconciliable. In such a case, one should
first normalize each component to unit variance first, i.e. b := b / 2.
The first principal component of a normalized vector would yield
[1 / sqrt(2), 1 / sqrt(2), 0], and would map [a, a, b] to sqrt(2) * a. The
second component would be [0, 0, 1] and map [a, a, b] to b. As can be seen,
the benefit of normalization is that PCA would capture highly correlated
components first and collapse them into a lower dimension.

#### Args:

* <b>`x`</b>: A rank-2 `Tensor`, 0th dim are rows, 1st dim are indices in row vectors.
* <b>`output_dim`</b>: The PCA output dimension (number of eigenvectors to return).
* <b>`dtype`</b>: Tensorflow dtype of entries in the returned matrix.
* <b>`name`</b>: (Optional) A name for this operation.


#### Raises:

* <b>`ValueError`</b>: if input is not a rank-2 Tensor.


#### Returns:

A 2D `Tensor` (matrix) M of shape (input_dim, output_dim).