<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.size" />
<meta itemprop="path" content="Stable" />
</div>

# tft.size

``` python
tft.size(
    x,
    reduce_instance_dims=True,
    name=None
)
```

Computes the total size of instances in a `Tensor` over the whole dataset.

#### Args:

* <b>`x`</b>: A `Tensor` or `SparseTensor`.
* <b>`reduce_instance_dims`</b>: By default collapses the batch and instance dimensions
    to arrive at a single scalar output. If False, only collapses the batch
    dimension and outputs a vector of the same shape as the input.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` of type int64.