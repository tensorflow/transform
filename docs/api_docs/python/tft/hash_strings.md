<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.hash_strings" />
<meta itemprop="path" content="Stable" />
</div>

# tft.hash_strings

``` python
tft.hash_strings(
    strings,
    hash_buckets,
    key=None,
    name=None
)
```

Hash strings into buckets.

#### Args:

* <b>`strings`</b>: a `Tensor` or `SparseTensor` of dtype `tf.string`.
* <b>`hash_buckets`</b>: the number of hash buckets.
* <b>`key`</b>: optional. An array of two Python `uint64`. If passed, output will be
    a deterministic function of `strings` and `key`. Note that hashing will be
    slower if this value is specified.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` or `SparseTensor` of dtype `tf.int64` with the same shape as the
input `strings`.


#### Raises:

* <b>`TypeError`</b>: if `strings` is not a `Tensor` or `SparseTensor` of dtype
  `tf.string`.