<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.sum" />
<meta itemprop="path" content="Stable" />
</div>

# tft.sum

``` python
tft.sum(
    x,
    reduce_instance_dims=True,
    name=None
)
```

Computes the sum of the values of a `Tensor` over the whole dataset.

#### Args:

* <b>`x`</b>: A `Tensor` or `SparseTensor`. Its type must be floating point
      (float{16|32|64}),integral (int{8|16|32|64}), or
      unsigned integral (uint{8|16})
* <b>`reduce_instance_dims`</b>: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a vector of the same shape as the input.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` containing the sum. If `x` is float32 or float64, the sum will
have the same type as `x`. If `x` is float16, the output is cast to float32.
If `x` is integral, the output is cast to [u]int64. If `x` is sparse and
reduce_inst_dims is False will return 0 in place where column has no values
across batches.


#### Raises:

* <b>`TypeError`</b>: If the type of `x` is not supported.