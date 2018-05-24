<!-- See: www.tensorflow.org/tfx/transform/ -->

# TensorFlow Transform Examples

## Simple example

There's a minimal TFX example available in the [GitHub repo](./simple_example.py).

## Census income example

The *Census income* dataset is provided by the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income).
This dataset contains both categorical and numeric data. See
[Get started with TensorFlow Transform](../get_started.md)
for details.

## Sentiment analysis example

Similar to the *Census income* example, but requires more extensive Apache Beam
processing before `tf.Transform` is invoked. See the
[sentiment analysis](./sentiment.md) for more information.
