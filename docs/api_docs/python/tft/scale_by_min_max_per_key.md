<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.scale_by_min_max_per_key" />
<meta itemprop="path" content="Stable" />
</div>

# tft.scale_by_min_max_per_key

``` python
tft.scale_by_min_max_per_key(
    x,
    key,
    output_min=0.0,
    output_max=1.0,
    elementwise=False,
    key_vocabulary_filename=None,
    name=None
)
```

Scale a numerical column into a predefined range on a per-key basis.

#### Args:

* <b>`x`</b>: A numeric `Tensor` or `SparseTensor`.
* <b>`key`</b>: A `Tensor` or `SparseTensor` of dtype tf.string.
      Must meet one of the following conditions:
      0. key is None
      1. Both x and key are dense,
      2. Both x and key are sparse and `key` must exactly match `x` in
         everything except values,
      3. The axis=1 index of each x matches its index of dense key.
* <b>`output_min`</b>: The minimum of the range of output values.
* <b>`output_max`</b>: The maximum of the range of output values.
* <b>`elementwise`</b>: If true, scale each element of the tensor independently.
* <b>`key_vocabulary_filename`</b>: (Optional) The file name for the per-key file.
    If None, this combiner will assume the keys fit in memory and will not
    store the analyzer result in a file. If '', a file name will be chosen
    based on the current TensorFlow scope. If not '', it should be unique
    within a given preprocessing function.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor`  or `SparseTensor` containing the input column scaled to
[output_min, output_max] on a per-key basis if a key is provided.


#### Raises:

* <b>`ValueError`</b>: If output_min, output_max have the wrong order.
* <b>`NotImplementedError`</b>: If elementwise is True and key is not None.
* <b>`InvalidArgumentError`</b>: If indices of sparse x and key do not match.