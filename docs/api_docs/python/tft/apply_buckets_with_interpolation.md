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

#### Args:

* <b>`x`</b>: A numeric input `Tensor` (tf.float32, tf.float64, tf.int32, tf.int64).
* <b>`bucket_boundaries`</b>: Sorted bucket boundaries as a rank-2 `Tensor`.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` of the same shape as `x`, normalized to the range [0, 1]. If the
  input x is tf.float64, the returned values will be tf.float64.
  Otherwise, returned values are tf.float32.