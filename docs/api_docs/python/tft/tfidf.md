<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.tfidf" />
<meta itemprop="path" content="Stable" />
</div>

# tft.tfidf

``` python
tft.tfidf(
    x,
    vocab_size,
    smooth=True,
    name=None
)
```

Maps the terms in x to their term frequency * inverse document frequency.

The term frequency of a term in a document is calculated as
(count of term in document) / (document size)

The inverse document frequency of a term is, by default, calculated as
1 + log((corpus size + 1) / (count of documents containing term + 1)).

Example usage:
  example strings [["I", "like", "pie", "pie", "pie"], ["yum", "yum", "pie]]
  in: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                            [1, 0], [1, 1], [1, 2]],
                   values=[1, 2, 0, 0, 0, 3, 3, 0])
  out: SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                    values=[1, 2, 0, 3, 0])
       SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                    values=[(1/5)*(log(3/2)+1), (1/5)*(log(3/2)+1), (3/5),
                            (2/3)*(log(3/2)+1), (1/3)]
  NOTE that the first doc's duplicate "pie" strings have been combined to
  one output, as have the second doc's duplicate "yum" strings.

#### Args:

* <b>`x`</b>: A `SparseTensor` representing int64 values (most likely that are the
      result of calling string_to_int on a tokenized string).
* <b>`vocab_size`</b>: An int - the count of vocab used to turn the string into int64s
      including any OOV buckets.
* <b>`smooth`</b>: A bool indicating if the inverse document frequency should be
      smoothed. If True, which is the default, then the idf is calculated as
      1 + log((corpus size + 1) / (document frequency of term + 1)).
      Otherwise, the idf is
      1 +log((corpus size) / (document frequency of term)), which could
      result in a division by zero error.
* <b>`name`</b>: (Optional) A name for this operation.


#### Returns:

Two `SparseTensor`s with indices [index_in_batch, index_in_bag_of_words].
The first has values vocab_index, which is taken from input `x`.
The second has values tfidf_weight.