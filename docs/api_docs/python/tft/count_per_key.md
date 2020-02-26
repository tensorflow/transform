<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.count_per_key" />
<meta itemprop="path" content="Stable" />
</div>

# tft.count_per_key

``` python
tft.count_per_key(
    key,
    key_vocabulary_filename=None,
    name=None
)
```

Computes the count of each element of a `Tensor`.

#### Args:

* <b>`key`</b>: A Tensor or `SparseTensor` of dtype tf.string or tf.int.
* <b>`key_vocabulary_filename`</b>: (Optional) The file name for the key-output mapping
    file. If None and key are provided, this combiner assumes the keys fit in
    memory and will not store the result in a file. If empty string, a file
    name will be chosen based on the current scope. If not an empty string,
    should be unique within a given preprocessing function.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

* <b>`Either`</b>:   (A) Two `Tensor`s: one the key vocab with dtype of input;
      the other the count for each key, dtype tf.int64. (if
      key_vocabulary_filename is None).
  (B) The filename where the key-value mapping is stored (if
      key_vocabulary_filename is not None).


#### Raises:

* <b>`TypeError`</b>: If the type of `x` is not supported.