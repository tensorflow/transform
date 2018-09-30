<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_buckets" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_buckets

``` python
tft.apply_buckets(
    x,
    bucket_boundaries,
    name=None
)
```

Returns a bucketized column, with a bucket index assigned to each input.

#### Args:

* <b>`x`</b>: A numeric input `Tensor` or `SparseTensor` whose values should be mapped
      to buckets.  For `SparseTensor`s, the non-missing values will be mapped
      to buckets and missing value left missing.
* <b>`bucket_boundaries`</b>: The bucket boundaries represented as a rank 1 `Tensor`.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` of the same shape as `x`, with each element in the
returned tensor representing the bucketized value. Bucketized value is
in the range [0, len(bucket_boundaries)].