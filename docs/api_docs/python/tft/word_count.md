<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.word_count" />
<meta itemprop="path" content="Stable" />
</div>

# tft.word_count

``` python
tft.word_count(
    tokens,
    name=None
)
```

Find the token count of each document/row.

`tokens` is either a `RaggedTensor` or `SparseTensor`, representing tokenized
strings. This function simply returns size of each row, so the dtype is not
constrained to string.

#### Args:

* <b>`tokens`</b>: either
    (1) a two-dimensional `SparseTensor`, or
    (2) a `RaggedTensor` with ragged rank of 1, non-ragged rank of 1
    of dtype `tf.string` containing tokens to be counted
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A one-dimensional `Tensor` the token counts of each row.


#### Raises:

* <b>`ValueError`</b>: if tokens is neither sparse nor ragged