<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.bucketize_per_key" />
<meta itemprop="path" content="Stable" />
</div>

# tft.bucketize_per_key

``` python
tft.bucketize_per_key(
    x,
    key,
    num_buckets,
    epsilon=None,
    name=None
)
```

Returns a bucketized column, with a bucket index assigned to each input.

#### Args:

* <b>`x`</b>: A numeric input `Tensor` or `SparseTensor` with rank 1, whose values
    should be mapped to buckets.  `SparseTensor`s will have their non-missing
    values mapped and missing values left as missing.
* <b>`key`</b>: A Tensor with the same shape as `x` and dtype tf.string.  If `x` is
    a `SparseTensor`, `key` must exactly match `x` in everything except
    values, i.e. indices and dense_shape must be identical.
* <b>`num_buckets`</b>: Values in the input `x` are divided into approximately
    equal-sized buckets, where the number of buckets is num_buckets.
* <b>`epsilon`</b>: (Optional) see `bucketize`
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` of the same shape as `x`, with each element in the
returned tensor representing the bucketized value. Bucketized value is
in the range [0, actual_num_buckets).


#### Raises:

* <b>`ValueError`</b>: If value of num_buckets is not > 1.