<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_combiner" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_combiner

``` python
tft.apply_combiner(
    inputs,
    output_dtypes,
    output_shapes,
    combiner,
    name
)
```

Applies the combiner over the whole dataset.

#### Args:

* <b>`inputs`</b>: A list of input `Tensor`s or `SparseTensor`s.
* <b>`output_dtypes`</b>: The list of dtypes of the output of the analyzer.
* <b>`output_shapes`</b>: The list of shapes of the output of the analyzer.  Must have
    the same length as output_dtypes.
* <b>`combiner`</b>: An object implementing the Combiner interface.
* <b>`name`</b>: Similar to a TF op name.  Used to define a unique scope for this
    analyzer, which can be used for debugging info.


#### Returns:

A list of `Tensor`s representing the combined values.  These will have
    `dtype` and `shape` given by `output_dtypes` and `output_shapes`.  These
    dtypes and shapes must be compatible with the combiner.


#### Raises:

* <b>`ValueError`</b>: If output_dtypes and output_shapes have different lengths.