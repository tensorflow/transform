<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.ptransform_analyzer" />
<meta itemprop="path" content="Stable" />
</div>

# tft.ptransform_analyzer

``` python
tft.ptransform_analyzer(
    inputs,
    output_dtypes,
    output_shapes,
    ptransform,
    name=None
)
```

Applies a user-provided PTransform over the whole dataset.

WARNING: This is experimental.

Note that in order to have asset files copied correctly, any outputs that
represent asset filenames must be added to the `tf.GraphKeys.ASSET_FILEPATHS`
collection by the caller.

#### Args:

* <b>`inputs`</b>: A list of input `Tensor`s.
* <b>`output_dtypes`</b>: The list of dtypes of the output of the analyzer.
* <b>`output_shapes`</b>: The list of shapes of the output of the analyzer.  Must have
    the same length as output_dtypes.
* <b>`ptransform`</b>: A Beam PTransform that accepts a Beam PCollection where each
    element is a list of `ndarray`s.  Each element in the list contains a
    batch of values for the corresponding input tensor of the analyzer.  It
    returns a tuple of `PCollection`, each containing a single element which
    is an `ndarray`.
* <b>`name`</b>: (Optional) Similar to a TF op name.  Used to define a unique scope for
    this analyzer, which can be used for debugging info.


#### Returns:

A list of output `Tensor`s.  These will have `dtype` and `shape` as
  specified by `output_dtypes` and `output_shapes`.


#### Raises:

* <b>`ValueError`</b>: If output_dtypes and output_shapes have different lengths.