<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.deduplicate_tensor_per_row" />
<meta itemprop="path" content="Stable" />
</div>

# tft.deduplicate_tensor_per_row

``` python
tft.deduplicate_tensor_per_row(
    input_tensor,
    name=None
)
```

Deduplicates each row (0-th dimension) of the provided tensor.

#### Args:

* <b>`input_tensor`</b>: A two-dimensional `Tensor` or `SparseTensor`. The first
    dimension is assumed to be the batch or "row" dimension, and deduplication
    is done on the 2nd dimension. If the Tensor is 1D it is returned as the
    equivalent `SparseTensor` since the "row" is a scalar can't be further
    deduplicated.
* <b>`name`</b>: Optional name for the operation.


#### Returns:

A  `SparseTensor` containing the unique set of values from each
  row of the input. Note: the original order of the input may not be
  preserved.