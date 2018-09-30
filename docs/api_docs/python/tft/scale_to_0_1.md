<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.scale_to_0_1" />
<meta itemprop="path" content="Stable" />
</div>

# tft.scale_to_0_1

``` python
tft.scale_to_0_1(
    x,
    elementwise=False,
    name=None
)
```

Returns a column which is the input column scaled to have range [0,1].

#### Args:

* <b>`x`</b>: A numeric `Tensor`.
* <b>`elementwise`</b>: If true, scale each element of the tensor independently.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` containing the input column scaled to [0, 1].