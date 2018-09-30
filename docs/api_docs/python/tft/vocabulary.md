<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.vocabulary" />
<meta itemprop="path" content="Stable" />
</div>

# tft.vocabulary

``` python
tft.vocabulary(
    x,
    top_k=None,
    frequency_threshold=None,
    vocab_filename=None,
    store_frequency=False,
    weights=None,
    labels=None,
    name=None
)
```

Computes the unique values of a `Tensor` over the whole dataset.

Computes The unique values taken by `x`, which can be a `Tensor` or
`SparseTensor` of any size.  The unique values will be aggregated over all
dimensions of `x` and all instances.

In case one of the tokens contains the '\n' or '\r' characters or is empty it
will be discarded since we are currently writing the vocabularies as text
files. This behavior will likely be fixed/improved in the future.

The unique values are sorted by decreasing frequency and then reverse
lexicographical order (e.g. [('a', 5), ('c', 3), ('b', 3)]).

For large datasets it is highly recommended to either set frequency_threshold
or top_k to control the size of the output, and also the run time of this
operation.

When labels are provided, we filter the vocabulary based on how correlated the
unique value is with a positive label (Mutual Information).

#### Args:

* <b>`x`</b>: An input `Tensor` or `SparseTensor` with dtype tf.string.
* <b>`top_k`</b>: Limit the generated vocabulary to the first `top_k` elements. If set
    to None, the full vocabulary is generated.
* <b>`frequency_threshold`</b>: Limit the generated vocabulary only to elements whose
    absolute frequency is >= to the supplied threshold. If set to None, the
    full vocabulary is generated.  Absolute frequency means the number of
    occurences of the element in the dataset, as opposed to the proportion of
    instances that contain that element.
* <b>`vocab_filename`</b>: The file name for the vocabulary file. If none, the
    "uniques" scope name in the context of this graph will be used as the file
    name. If not None, should be unique within a given preprocessing function.
    NOTE To make your pipelines resilient to implementation details please
    set `vocab_filename` when you are using the vocab_filename on a downstream
    component.
* <b>`store_frequency`</b>: If True, frequency of the words is stored in the
    vocabulary file. In the case labels are provided, the mutual
    information is stored in the file instead. Each line in the file
    will be of the form 'frequency word'.
* <b>`weights`</b>: (Optional) Weights `Tensor` for the vocabulary. It must have the
    same shape as x.
* <b>`labels`</b>: (Optional) Labels `Tensor` for the vocabulary. It must have dtype
    int64, have values 0 or 1, and have the same shape as x.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

The path name for the vocabulary file containing the unique values of `x`.


#### Raises:

* <b>`ValueError`</b>: If `top_k` or `frequency_threshold` is negative.