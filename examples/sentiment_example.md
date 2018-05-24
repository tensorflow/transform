<!-- See: www.tensorflow.org/tfx/transform/ -->

# More Examples with tf.Transform

We provide some more examples which show more advanced kinds of preprocessing.

## Sentiment Analysis

The example `sentiment_example.py` uses the
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
which contains 50000 movie reviews equally split into train and test sets. To run
this example, the data should be downloaded and unzipped to a directory. As with
`census_example.py`, when running `sentiment_example.py`, the directory
containing the data should be passed as the first argument. The script will
create a temp subdirectory of the data directory to put preprocessed data in.

This example is similar to the `census_example.py` example but there are some
differences. More extensive Beam processing is required before tf.Transform is
invoked. In this example, the data must be read from multiple files across
separate directories for positive and negative examples. These datasets then
have the correct label attached and are then shuffled.

Because the data comes in the form of separate files for each review, with
separate directories for positive and negative reviews, this example does not
use a tf.Transform provided coder to produce the data that is read by
tf.Transform. Instead, this data is produced directly by the Beam pipeline, in
the form of a PCollection of dictionaries. Note that the format of these
dictionaries is not yet specified in the API and may change. Updates will be
made to the documentation as the format is finalized.

The preprocessing done by tf.Transform is also more complex. Unlike the census
dataset example, here the data comes in the form of a single feature which is
the full text of a movie review. This is split into sentences using the
`tf.string_split` function. The `tf.string_split` function takes a rank 1 tensor
and converts it to a rank 2 `SparseTensor` containing the individual tokens.
This `SparseTensor` is then converted to a `SparseTensor` of `int64`s with the
same shape, using `tft.string_to_int`.

During the training and evaluation phase, the `SparseTensor` representing the
tokenized, integerized review text, is used as the input to a bag-of-words
model. In particular, the tensor is wrapped as a `FeatureColumn` which (like the
census example) is created with `sparse_column_with_integerized_feature`.
However in this case, instead of a length 1 vector (per instance) we have a
vector whose size is the number of tokens. In this circumstance, the vector of
integerized tokens is interpreted as a bag-of-words.

The bag-of-words model gives an accuracy of 87.6% which is close to the accuracy
of the bag-of-words model for the same dataset given by Maas et al. (2011). Note
that their bag of words model may not be identical to ours so we do not expect
exact agreement.

# References

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and
Christopher Potts. (2011).
[Learning Word Vectors for Sentiment Analysis.](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) *The 49th
Annual Meeting of the Association for Computational Linguistics (ACL 2011).*
