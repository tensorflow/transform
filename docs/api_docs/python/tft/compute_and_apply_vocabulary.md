<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.compute_and_apply_vocabulary" />
<meta itemprop="path" content="Stable" />
</div>

# tft.compute_and_apply_vocabulary

``` python
tft.compute_and_apply_vocabulary(
    x,
    default_value=-1,
    top_k=None,
    frequency_threshold=None,
    num_oov_buckets=0,
    vocab_filename=None,
    weights=None,
    labels=None,
    name=None
)
```

Generates a vocabulary for `x` and maps it to an integer with this vocab.

In case one of the tokens contains the '\n' or '\r' characters or is empty it
will be discarded since we are currently writing the vocabularies as text
files. This behavior will likely be fixed/improved in the future.

Note that this function will cause a vocabulary to be computed.  For large
datasets it is highly recommended to either set frequency_threshold or top_k
to control the size of the vocabulary, and also the run time of this
operation.

#### Args:

* <b>`x`</b>: A `Tensor` or `SparseTensor` of type tf.string.
* <b>`default_value`</b>: The value to use for out-of-vocabulary values, unless
    'num_oov_buckets' is greater than zero.
* <b>`top_k`</b>: Limit the generated vocabulary to the first `top_k` elements. If set
    to None, the full vocabulary is generated.
* <b>`frequency_threshold`</b>: Limit the generated vocabulary only to elements whose
    absolute frequency is >= to the supplied threshold. If set to None, the
    full vocabulary is generated.  Absolute frequency means the number of
    occurences of the element in the dataset, as opposed to the proportion of
    instances that contain that element.
* <b>`num_oov_buckets`</b>:  Any lookup of an out-of-vocabulary token will return a
    bucket ID based on its hash if `num_oov_buckets` is greater than zero.
    Otherwise it is assigned the `default_value`.
* <b>`vocab_filename`</b>: The file name for the vocabulary file. If None, a name based
    on the scope name in the context of this graph will be used as the
    file name. If not None, should be unique within a given preprocessing
    function.
    NOTE in order to make your pipelines resilient to implementation details
    please set `vocab_filename` when you are using the vocab_filename on a
    downstream component.
* <b>`weights`</b>: (Optional) Weights `Tensor` for the vocabulary. It must have the
    same shape as x.
* <b>`labels`</b>: (Optional) Labels `Tensor` for the vocabulary. It must have dtype
    int64, have values 0 or 1, and have the same shape as x.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `Tensor` or `SparseTensor` where each string value is mapped to an
integer. Each unique string value that appears in the vocabulary
is mapped to a different integer and integers are consecutive starting from
zero. String value not in the vocabulary is assigned default_value.


#### Raises:

* <b>`ValueError`</b>: If `top_k` or `frequency_threshold` is negative.
