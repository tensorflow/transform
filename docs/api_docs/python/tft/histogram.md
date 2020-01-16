<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.histogram" />
<meta itemprop="path" content="Stable" />
</div>

# tft.histogram

``` python
tft.histogram(
    x,
    boundaries=None,
    categorical=False,
    name=None
)
```

Computes a histogram over x, given the bin boundaries or bin count.

Ex (1):
counts, boundaries = histogram([0, 1, 0, 1, 0, 3, 0, 1], range(5))
counts: [4, 3, 0, 1, 0]
boundaries: [0, 1, 2, 3, 4]

Ex (2):
Can be used to compute class weights.
counts, classes = histogram([0, 1, 0, 1, 0, 3, 0, 1], categorical=True)
probabilities = counts / tf.reduce_sum(counts)
class_weights = dict(map(lambda (a, b): (a.numpy(), 1.0 / b.numpy()),
                         zip(classes, probabilities)))

#### Args:

* <b>`x`</b>: A `Tensor` or `SparseTensor`.
* <b>`boundaries`</b>: (Optional) A `Tensor` or `int` used to build the histogram;
      ignored if `categorical` is True. If possible, provide boundaries as
      multiple sorted values.  Default to 10 intervals over the 0-1 range,
      or find the min/max if an int is provided (not recommended because
      multi-phase analysis is inefficient).
* <b>`categorical`</b>: (Optional) A `bool` that treats `x` as discrete values if true.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

* <b>`counts`</b>: The histogram, as counts per bin.
* <b>`boundaries`</b>: A `Tensor` used to build the histogram representing boundaries.