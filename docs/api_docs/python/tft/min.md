<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.min" />
<meta itemprop="path" content="Stable" />
</div>

# tft.min

``` python
tft.min(
    x,
    reduce_instance_dims=True,
    name=None
)
```

Computes the minimum of the values of a `Tensor` over the whole dataset.

In the case of a `SparseTensor` missing values will be used in return value:
for float, NaN is used and for other dtypes the max is used.

#### Args:

* <b>`x`</b>: A `Tensor` or `SparseTensor`.
* <b>`reduce_instance_dims`</b>: By default collapses the batch and instance dimensions
    to arrive at a single scalar output. If False, only collapses the batch
    dimension and outputs a `Tensor` of the same shape as the input.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` with the same type as `x`.


#### Raises:

* <b>`TypeError`</b>: If the type of `x` is not supported.