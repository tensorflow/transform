<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.scale_to_0_1_per_key" />
<meta itemprop="path" content="Stable" />
</div>

# tft.scale_to_0_1_per_key

``` python
tft.scale_to_0_1_per_key(
    x,
    key,
    elementwise=False,
    key_vocabulary_filename=None,
    name=None
)
```

Returns a column which is the input column scaled to have range [0,1].

#### Args:

* <b>`x`</b>: A numeric `Tensor`.
* <b>`key`</b>: A `Tensor` of type string.
* <b>`elementwise`</b>: If true, scale each element of the tensor independently.
* <b>`key_vocabulary_filename`</b>: (Optional) The file name for the per-key file.
    If None, this combiner will assume the keys fit in memory and will not
    store the analyzer result in a file. If '', the "uniques" scope name in
    the context of this graph will be used as the file name. If not '', it
    should be unique within a given preprocessing function.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` containing the input column scaled to [0, 1], per key.