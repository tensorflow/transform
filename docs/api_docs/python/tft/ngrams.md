<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.ngrams" />
<meta itemprop="path" content="Stable" />
</div>

# tft.ngrams

``` python
tft.ngrams(
    tokens,
    ngram_range,
    separator,
    name=None
)
```

Create a `SparseTensor` of n-grams.

Given a `SparseTensor` of tokens, returns a `SparseTensor` containing the
ngrams that can be constructed from each row.

`separator` is inserted between each pair of tokens, so " " would be an
appropriate choice if the tokens are words, while "" would be an appropriate
choice if they are characters.

Example:

`tokens` is a `SparseTensor` with

indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]]
values = ['One', 'was', 'Johnny', 'Two', 'was', 'a', 'rat']
dense_shape = [2, 4]

If we set
ngrams_range = (1,3)
separator = ' '

output is a `SparseTensor` with

indices = [[0, 0], [0, 1], [0, 2], ..., [1, 6], [1, 7], [1, 8]]
values = ['One', 'One was', 'One was Johnny', 'was', 'was Johnny', 'Johnny',
          'Two', 'Two was', 'Two was a', 'was', 'was a', 'was a rat', 'a',
          'a rat', 'rat']
dense_shape = [2, 9]

#### Args:

* <b>`tokens`</b>: a two-dimensional`SparseTensor` of dtype `tf.string` containing
    tokens that will be used to construct ngrams.
* <b>`ngram_range`</b>: A pair with the range (inclusive) of ngram sizes to return.
* <b>`separator`</b>: a string that will be inserted between tokens when ngrams are
    constructed.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

A `SparseTensor` containing all ngrams from each row of the input.


#### Raises:

* <b>`ValueError`</b>: if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]