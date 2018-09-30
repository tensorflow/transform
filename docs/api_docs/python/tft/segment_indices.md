<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.segment_indices" />
<meta itemprop="path" content="Stable" />
</div>

# tft.segment_indices

``` python
tft.segment_indices(
    segment_ids,
    name=None
)
```

Returns a `Tensor` of indices within each segment.

segment_ids should be a sequence of non-decreasing non-negative integers that
define a set of segments, e.g. [0, 0, 1, 2, 2, 2] defines 3 segments of length
2, 1 and 3.  The return value is a `Tensor` containing the indices within each
segment.

Example input: [0, 0, 1, 2, 2, 2]
Example output: [0, 1, 0, 0, 1, 2]

#### Args:

* <b>`segment_ids`</b>: A 1-d `Tensor` containing an non-decreasing sequence of
      non-negative integers with type `tf.int32` or `tf.int64`.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` containing the indices within each segment.