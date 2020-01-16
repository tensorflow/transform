<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.estimated_probability_density" />
<meta itemprop="path" content="Stable" />
</div>

# tft.estimated_probability_density

``` python
tft.estimated_probability_density(
    x,
    boundaries=None,
    categorical=False,
    name=None
)
```

Computes an approximate probability density at each x, given the bins.

Using this type of fixed-interval method has several benefits compared to
  bucketization, although may not always be preferred.
  1. Quantiles does not work on categorical data.
  2. The quantiles algorithm does not currently operate on multiple features
  jointly, only independently.

Ex: Outlier detection in a multi-modal or arbitrary distribution.
  Imagine a value x where a simple model is highly predictive of a target y
  within certain densely populated ranges. Outside these ranges, we may want
  to treat the data differently, but there are too few samples for the model
  to detect them by case-by-case treatment.
  One option would be to use the density estimate for this purpose:

  outputs['x_density'] = tft.estimated_prob(inputs['x'], bins=100)
  outputs['outlier_x'] = tf.where(outputs['x_density'] < OUTLIER_THRESHOLD,
                                  tf.constant([1]), tf.constant([0]))

  This exercise uses a single variable for illustration, but a direct density
  metric would become more useful with higher dimensions.

Note that we normalize by average bin_width to arrive at a probability density
estimate. The result resembles a pdf, not the probability that a value falls
in the bucket (except in the categorical case).

#### Args:

* <b>`x`</b>: A `Tensor`.
* <b>`boundaries`</b>: (Optional) A `Tensor` or int used to approximate the density.
      If possible provide boundaries as a Tensor of multiple sorted values.
      Will default to 10 intervals over the 0-1 range, or find the min/max
      if an int is provided (not recommended because multi-phase analysis is
      inefficient). If the boundaries are known as potentially arbitrary
      interval boundaries, sizes are assumed to be equal. If the sizes are
      unequal, density may be inaccurate. Ignored if `categorical` is true.
* <b>`categorical`</b>: (Optional) A `bool` that will treat x as categorical if true.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` the same shape as x, the probability density estimate at x (or
probability mass estimate if `categorical` is True).


#### Raises:

* <b>`NotImplementedError`</b>: If `x` is SparseTensor.