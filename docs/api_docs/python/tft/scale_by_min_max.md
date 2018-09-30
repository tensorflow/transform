<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.scale_by_min_max" />
<meta itemprop="path" content="Stable" />
</div>

# tft.scale_by_min_max

``` python
tft.scale_by_min_max(
    x,
    output_min=0.0,
    output_max=1.0,
    elementwise=False,
    name=None
)
```

Scale a numerical column into the range [output_min, output_max].

#### Args:

* <b>`x`</b>: A numeric `Tensor`.
* <b>`output_min`</b>: The minimum of the range of output values.
* <b>`output_max`</b>: The maximum of the range of output values.
* <b>`elementwise`</b>: If true, scale each element of the tensor independently.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` containing the input column scaled to [output_min, output_max].


#### Raises:

* <b>`ValueError`</b>: If output_min, output_max have the wrong order.