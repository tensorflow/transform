<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.count_per_key" />
<meta itemprop="path" content="Stable" />
</div>

# tft.count_per_key

``` python
tft.count_per_key(
    key,
    name=None
)
```

Computes the count of each element of a `Tensor`.

#### Args:

* <b>`key`</b>: A Tensor or `SparseTensor` of dtype tf.string or tf.int.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

Two `Tensor`s: one the key vocab with dtype of input;
    the other the count for each key, dtype tf.int64.


#### Raises:

* <b>`TypeError`</b>: If the type of `x` is not supported.