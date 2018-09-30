<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.bucketize" />
<meta itemprop="path" content="Stable" />
</div>

# tft.bucketize

``` python
tft.bucketize(
    x,
    num_buckets,
    epsilon=None,
    weights=None,
    name=None
)
```

Returns a bucketized column, with a bucket index assigned to each input.

#### Args:

* <b>`x`</b>: A numeric input `Tensor` or `SparseTensor` whose values should be mapped
    to buckets.  For a `SparseTensor` only non-missing values will be included
    in the quantiles computation, and the result of `bucketize` will be a
    `SparseTensor` with non-missing values mapped to buckets.
* <b>`num_buckets`</b>: Values in the input `x` are divided into approximately
    equal-sized buckets, where the number of buckets is num_buckets.
    This is a hint. The actual number of buckets computed can be
    less or more than the requested number. Use the generated metadata to
    find the computed number of buckets.
* <b>`epsilon`</b>: (Optional) Error tolerance, typically a small fraction close to
    zero. If a value is not specified by the caller, a suitable value is
    computed based on experimental results.  For `num_buckets` less
    than 100, the value of 0.01 is chosen to handle a dataset of up to
    ~1 trillion input data values.  If `num_buckets` is larger,
    then epsilon is set to (1/`num_buckets`) to enforce a stricter
    error tolerance, because more buckets will result in smaller range for
    each bucket, and so we want the boundaries to be less fuzzy.
    See analyzers.quantiles() for details.
* <b>`weights`</b>: (Optional) Weights tensor for the quantiles. Tensor must have the
    same shape as x.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` of the same shape as `x`, with each element in the
returned tensor representing the bucketized value. Bucketized value is
in the range [0, actual_num_buckets). Sometimes the actual number of buckets
can be different than num_buckets hint, for example in case the number of
distinct values is smaller than num_buckets, or in cases where the
input values are not uniformly distributed.


#### Raises:

* <b>`ValueError`</b>: If value of num_buckets is not > 1.