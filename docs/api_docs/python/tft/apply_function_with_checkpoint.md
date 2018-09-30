<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_function_with_checkpoint" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_function_with_checkpoint

``` python
tft.apply_function_with_checkpoint(
    fn,
    inputs,
    checkpoint,
    include=None,
    exclude=None
)
```

Applies a tensor-in-tensor-out function with variables to some `Tensor`s.

Variable values are loaded from the given checkpoint path. Note that the
input_tensor_func, together with the checkpoint, will be converted to an
all-constants graph, so ops requiring graph collections, such as table lookup
(which requires a table init op being added to TABLE_INITIALIZERS collection),
are not supported.

#### Args:

* <b>`fn`</b>: A tensor-in-tensor-out function that may contain variables.
* <b>`inputs`</b>: A list of `Tensor`s to apply `fn` to.
* <b>`checkpoint`</b>: The checkpoint path to load variables from.
* <b>`include`</b>: An optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to include. If None, all
      variables will be included.
* <b>`exclude`</b>: An optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to exclude. If None, no
      variables will be excluded.


#### Returns:

A `Tensor` or list of `Tensor`s representing the application of `fn`.


#### Raises:

* <b>`ValueError`</b>: if the input tensor-in-tensor-out function adds to
      TABLE_INITIALIZERS collections.