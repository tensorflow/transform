<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.var" />
<meta itemprop="path" content="Stable" />
</div>

# tft.var

``` python
tft.var(
    x,
    reduce_instance_dims=True,
    name=None,
    output_dtype=None
)
```

Computes the variance of the values of a `Tensor` over the whole dataset.

Uses the biased variance (0 delta degrees of freedom), as given by
(x - mean(x))**2 / length(x).

#### Args:

* <b>`x`</b>: `Tensor` or `SparseTensor`. Its type must be floating point
      (float{16|32|64}), or integral ([u]int{8|16|32|64}).
* <b>`reduce_instance_dims`</b>: By default collapses the batch and instance dimensions
      to arrive at a single scalar output. If False, only collapses the batch
      dimension and outputs a vector of the same shape as the input.
* <b>`name`</b>: (Optional) A name for this operation.
* <b>`output_dtype`</b>: (Optional) If not None, casts the output tensor to this type.


#### Returns:

A `Tensor` containing the variance. If `x` is floating point, the variance
will have the same type as `x`. If `x` is integral, the output is cast to
float32.


#### Raises:

* <b>`TypeError`</b>: If the type of `x` is not supported.