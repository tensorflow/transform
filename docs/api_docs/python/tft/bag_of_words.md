<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.bag_of_words" />
<meta itemprop="path" content="Stable" />
</div>

# tft.bag_of_words

``` python
tft.bag_of_words(
    tokens,
    ngram_range,
    separator,
    name=None
)
```

Computes a bag of "words" based on the specified ngram configuration.

A light wrapper around tft.ngrams. First computes ngrams, then transforms the
ngram representation (list semantics) into a Bag of Words (set semantics) per
row. Each row reflects the set of *unique* ngrams present in an input record.

See tft.ngrams for more information.

#### Args:

* <b>`tokens`</b>: a two-dimensional `SparseTensor` of dtype `tf.string` containing
    tokens that will be used to construct a bag of words.
* <b>`ngram_range`</b>: A pair with the range (inclusive) of ngram sizes to compute.
* <b>`separator`</b>: a string that will be inserted between tokens when ngrams are
    constructed.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `SparseTensor` containing the unique set of ngrams from each row of the
  input. Note: the original order of the ngrams may not be preserved.