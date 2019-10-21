<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_buckets_with_interpolation" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_buckets_with_interpolation

``` python
tft.apply_buckets_with_interpolation(
    x,
    bucket_boundaries,
    name=None
)
```

Interpolates within the provided buckets and then normalizes to 0 to 1.

A method for normalizing continuous numeric data to the range [0, 1].
Numeric values are first bucketized according to the provided boundaries, then
linearly interpolated within their respective bucket ranges. Finally, the
interpolated values are normalized to the range [0, 1]. Values that are
less than or equal to the lowest boundary, or greater than or equal to the
highest boundary, will be mapped to 0 and 1 respectively.

This is a non-linear approach to normalization that is less sensitive to
outliers than min-max or z-score scaling. When outliers are present, standard
forms of normalization can leave the majority of the data compressed into a
very small segment of the output range, whereas this approach tends to spread
out the more frequent values (if quantile buckets are used). Note that
distance relationships in the raw data are not necessarily preserved (data
points that close to each other in the raw feature space may not be equally
close in the transformed feature space). This means that unlike linear
normalization methods, correlations between features may be distorted by the
transformation. This scaling method may help with stability and minimize
exploding gradients in neural networks.

#### Args:

* <b>`x`</b>: A numeric input `Tensor`/`SparseTensor` (tf.float[32|64], tf.int[32|64])
* <b>`bucket_boundaries`</b>: Sorted bucket boundaries as a rank-2 `Tensor`.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` or `SparseTensor` of the same shape as `x`, normalized to the
  range [0, 1]. If the input x is tf.float64, the returned values will be
  tf.float64. Otherwise, returned values are tf.float32.