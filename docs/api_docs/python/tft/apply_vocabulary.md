<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_vocabulary" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_vocabulary

``` python
tft.apply_vocabulary(
    x,
    deferred_vocab_filename_tensor,
    default_value=-1,
    num_oov_buckets=0,
    lookup_fn=None,
    name=None
)
```

Maps `x` to a vocabulary specified by the deferred tensor.

This function also writes domain statistics about the vocabulary min and max
values. Note that the min and max are inclusive, and depend on the vocab size,
num_oov_buckets and default_value.

In case one of the tokens contains the '\n' or '\r' characters or is empty it
will be discarded since we are currently writing the vocabularies as text
files. This behavior will likely be fixed/improved in the future.

#### Args:

* <b>`x`</b>: A `Tensor` or `SparseTensor` of type tf.string to which the vocabulary
    transformation should be applied.
    The column names are those intended for the transformed tensors.
* <b>`deferred_vocab_filename_tensor`</b>: The deferred vocab filename tensor as
    returned by <a href="../tft/vocabulary.md"><code>tft.vocabulary</code></a>.
* <b>`default_value`</b>: The value to use for out-of-vocabulary values, unless
    'num_oov_buckets' is greater than zero.
* <b>`num_oov_buckets`</b>:  Any lookup of an out-of-vocabulary token will return a
    bucket ID based on its hash if `num_oov_buckets` is greater than zero.
    Otherwise it is assigned the `default_value`.
* <b>`lookup_fn`</b>: Optional lookup function, if specified it should take a tensor
    and a deferred vocab filename as an input and return a lookup `op` along
    with the table size, by default `apply_vocab` performs a
    lookup_ops.index_table_from_file for the table lookup.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` or `SparseTensor` where each string value is mapped to an
integer. Each unique string value that appears in the vocabulary
is mapped to a different integer and integers are consecutive
starting from zero, and string value not in the vocabulary is
assigned default_value.