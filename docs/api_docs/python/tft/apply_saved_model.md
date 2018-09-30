<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_saved_model" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_saved_model

``` python
tft.apply_saved_model(
    model_dir,
    inputs,
    tags,
    signature_name=None,
    output_keys_in_signature=None
)
```

Applies a SavedModel to some `Tensor`s.

Applies a SavedModel to `inputs`. The SavedModel is specified with
`model_dir`, `tags` and `signature_name`. Note that the SavedModel will be
converted to an all-constants graph.

#### Args:

* <b>`model_dir`</b>: A path containing a SavedModel.
* <b>`inputs`</b>: A dict whose keys are the names from the input signature and whose
      values are `Tensor`s. If there is only one input in the model's input
      signature then `inputs` can be a single `Tensor`.
* <b>`tags`</b>: The tags specifying which metagraph to load from the SavedModel.
* <b>`signature_name`</b>: Specify signature of the loaded model. The default value
      None can be used if there is only one signature in the MetaGraphDef.
* <b>`output_keys_in_signature`</b>: A list of strings which should be a subset of
      the outputs in the signature of the SavedModel. The returned `Tensor`s
      will correspond to specified output `Tensor`s, in the same order. The
      default value None can be used if there is only one output from
      signature.


#### Returns:

A `Tensor` or list of `Tensor`s representing the application of the
    SavedModel.


#### Raises:

* <b>`ValueError`</b>: if
  `inputs` is invalid type, or
  `signature_name` is None but the SavedModel contains multiple signature, or
  `inputs` do not match the signature inputs, or
  `output_keys_in_signature` is not a subset of the signature outputs.