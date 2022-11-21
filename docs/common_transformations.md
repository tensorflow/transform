# Common Transformations

[TOC]

In this document we describe how to do common transformations with tf.transform.

We assume you have already constructed the beam pipeline along the lines of the
examples, and only describe what needs to be added to `preprocessing_fn` and
possibly model.

## Using String/Categorical data

The following `preprocessing_fn` will compute a vocabulary over the values of
feature `x` with tokens in descending frequency order, convert feature `x`
values to their index in the vocabulary, and finally perform a one-hot encoding
for the output.

This is common for example in use cases where the label feature is a categorical
string.
The resulting one-hot encoding is ready for training.

Note: this example produces `x_out` as a potentially large dense tensor. This is
fine as long as the transformed data doesn't get materialized, and this is the
format expected in training. Otherwise, a more efficient representation would be
a `tf.SparseTensor`, in which case only a single index and value (1) is used to
represent each instance.

```python
def preprocessing_fn(inputs):
  integerized = tft.compute_and_apply_vocabulary(
      inputs['x'],
      num_oov_buckets=1,
      vocab_filename='x_vocab')
  one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(tft.experimental.get_vocabulary_size_by_name('x_vocab') + 1,
                    tf.int32),
      on_value=1.0,
      off_value=0.0)
  return {
    'x_out': one_hot_encoded,
  }
```

## Mean imputation for missing data

In this example, feature `x` is an optional feature, represented as a
`tf.SparseTensor` in the `preprocessing_fn`. In order to convert it to a dense
tensor, we compute its mean, and set the mean to be the default value when it
is missing from an instance.

The resulting dense tensor will have the shape `[None, 1]`, `None` represents
the batch dimension, and for the second dimension it will be the number of
values that `x` can have per instance. In this case it's 1.

```python
def preprocessing_fn(inputs):
  return {
      'x_out': tft.sparse_tensor_to_dense_with_shape(
          inputs['x'], default_value=tft.mean(x), shape=[None, 1])
  }
```
