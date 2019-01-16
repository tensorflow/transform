<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.quantiles" />
<meta itemprop="path" content="Stable" />
</div>

# tft.quantiles

``` python
tft.quantiles(
    x,
    num_buckets,
    epsilon,
    weights=None,
    name=None
)
```

Computes the quantile boundaries of a `Tensor` over the whole dataset.

quantile boundaries are computed using approximate quantiles,
and error tolerance is specified using `epsilon`. The boundaries divide the
input tensor into approximately equal `num_buckets` parts.
See go/squawd for details, and how to control the error due to approximation.

#### Args:

* <b>`x`</b>: An input `Tensor`.
* <b>`num_buckets`</b>: Values in the `x` are divided into approximately equal-sized
    buckets, where the number of buckets is num_buckets. This is a hint. The
    actual number of buckets computed can be less or more than the requested
    number. Use the generated metadata to find the computed number of buckets.
* <b>`epsilon`</b>: Error tolerance, typically a small fraction close to zero (e.g.
    0.01). Higher values of epsilon increase the quantile approximation, and
    hence result in more unequal buckets, but could improve performance,
    and resource consumption.  Some measured results on memory consumption:
      For epsilon = 0.001, the amount of memory for each buffer to hold the
      summary for 1 trillion input values is ~25000 bytes. If epsilon is
      relaxed to 0.01, the buffer size drops to ~2000 bytes for the same input
      size. If we use a strict epsilon value of 0, the buffer size is same
      size as the input, because the intermediate stages have to remember
      every input and the quantile boundaries can be found only after an
      equivalent to a full sorting of input. The buffer size also determines
      the amount of work in the different stages of the beam pipeline, in
      general, larger epsilon results in fewer and smaller stages, and less
      time. For more performance
      trade-offs see also http://web.cs.ucla.edu/~weiwang/paper/SSDBM07_2.pdf
* <b>`weights`</b>: (Optional) Weights tensor for the quantiles. Tensor must have the
    same shape as x.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

The bucket boundaries represented as a list, with num_bucket-1 elements
See code below for discussion on the type of bucket boundaries.