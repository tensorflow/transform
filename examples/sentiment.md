<!-- See: www.tensorflow.org/tfx/transform/ -->

# Sentiment Analysis

[`sentiment_example.py`](./sentiment_example.py)
uses the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
and contains 50,000 movie reviews equally split into train and test sets. To run
this example, download and unzip the data set to a directory. Pass this
directory as an argument to `sentiment_example.py`. The script creates a
temporary sub-directory to add the preprocessed data.

This example is similar to the
[Census income example](../get_started.md) but
requires more extensive Apache Beam processing before invoking `tf.Transform`.
Here, the data must be read from multiple files across separate directories for
positive and negative examples. Then, the correct labels are attached to the
dataset and shuffled.

Since the input data uses separate files for each review (with separate
directories for positive and negative reviews), this example does not use a
coder provided by `tf.Transform` to produce data read by `tf.Transform`.
Instead, the data is produced directly by the Beam pipeline in the form of a
`PCollection` of dictionaries.

Caution: The format of these dictionaries is not specified in the API and may
change. When the format is specified, the documentation will be updated.

The `tf.Transform` preprocessing is more complex. Unlike the Census income
example, the data in this example uses a single feature for the full text of a
movie review. This is split into sentences using the `tf.string_split`
function. The `tf.string_split` function takes a rank 1 tensor and converts it
to a rank 2 `SparseTensor` that contains the individual tokens. Then, using
`tft.compute_and_apply_vocabulary`, this `SparseTensor` is converted to a
`SparseTensor` of `int64`s with the same shape.

During the training and evaluation phase, the `SparseTensor` that represents
the review text (tokenized and integerized) is used as the input to a
bag-of-words model. In particular, the tensor is wrapped as a `FeatureColumn`
created with `sparse_column_with_integerized_feature`. However, instead
of a vector with a length of `1` (per instance), there's a vector with the length
of the number of tokens. In this circumstance, the vector of integerized tokens
is interpreted as a bag-of-words.

The bag-of-words model gives an accuracy of 87.6%—close to the accuracy of the
bag-of-words model for the same dataset given by
[Maas et al. (2011)](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf).
But since their bag-of-words model may not be identical to ours, we do not expect
exact agreement.
