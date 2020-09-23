<!-- See: www.tensorflow.org/tfx/transform/ -->

# TensorFlow Transform

[![Python](https://img.shields.io/pypi/pyversions/tensorflow-transform.svg?style=plastic)](https://github.com/tensorflow/transform)
[![PyPI](https://badge.fury.io/py/tensorflow-transform.svg)](https://badge.fury.io/py/tensorflow-transform)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)

*TensorFlow Transform* is a library for preprocessing data with TensorFlow.
`tf.Transform` is useful for data that requires a full-pass, such as:

* Normalize an input value by mean and standard deviation.
* Convert strings to integers by generating a vocabulary over all input values.
* Convert floats to integers by assigning them to buckets based on the observed
  data distribution.

TensorFlow has built-in support for manipulations on a single example or a batch
of examples. `tf.Transform` extends these capabilities to support full-passes
over the example data.

The output of `tf.Transform` is exported as a
[TensorFlow graph](http://tensorflow.org/guide/graphs) to use for training and serving.
Using the same graph for both training and serving can prevent skew since the
same transformations are applied in both stages.

For an introduction to `tf.Transform`, see the `tf.Transform` section of the
TFX Dev Summit talk on TFX
([link](https://www.youtube.com/watch?v=vdG7uKQ2eKk&feature=youtu.be&t=199)).

Caution: `tf.Transform` may be backwards incompatible before version 1.0.

## Installation

The `tensorflow-transform`
[PyPI package](https://pypi.org/project/tensorflow-transform/) is the
recommended way to install `tf.Transform`:

```bash
pip install tensorflow-transform
```

### Notable Dependencies

TensorFlow is required.

[Apache Beam](https://beam.apache.org/) is required; it's the way that efficient
distributed computation is supported. By default, Apache Beam runs in local
mode but can also run in distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/) and other Apache
Beam
[runners](https://beam.apache.org/documentation/runners/capability-matrix/).

[Apache Arrow](https://arrow.apache.org/) is also required. TFT uses Arrow to
represent data internally in order to make use of vectorized numpy functions.

## Compatible versions

The following table is the `tf.Transform` package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

tensorflow-transform                                                            | apache-beam[gcp] | tensorflow        | tensorflow-metadata | tfx-bsl |
------------------------------------------------------------------------------- | -----------------| ------------------|---------------------|---------|
[GitHub master](https://github.com/tensorflow/transform/blob/master/RELEASE.md) | 2.24.0           | nightly (1.x/2.x) | 0.24.0              | 0.24.1  |
[0.24.1](https://github.com/tensorflow/transform/blob/v0.24.1/RELEASE.md)       | 2.24.0           | 1.15 / 2.3        | 0.24.0              | 0.24.1  |
[0.24.0](https://github.com/tensorflow/transform/blob/v0.24.0/RELEASE.md)       | 2.23.0           | 1.15 / 2.3        | 0.24.0              | 0.24.0  |
[0.23.0](https://github.com/tensorflow/transform/blob/v0.23.0/RELEASE.md)       | 2.23.0           | 1.15 / 2.3        | 0.23.0              | 0.23.0  |
[0.22.0](https://github.com/tensorflow/transform/blob/v0.22.0/RELEASE.md)       | 2.20.0           | 1.15 / 2.2        | 0.22.0              | 0.22.0  |
[0.21.2](https://github.com/tensorflow/transform/blob/v0.21.2/RELEASE.md)       | 2.17.0           | 1.15 / 2.1        | 0.21.0              | 0.21.3  |
[0.21.0](https://github.com/tensorflow/transform/blob/v0.21.0/RELEASE.md)       | 2.17.0           | 1.15 / 2.1        | 0.21.0              | 0.21.0  |
[0.15.0](https://github.com/tensorflow/transform/blob/v0.15.0/RELEASE.md)       | 2.16.0           | 1.15 / 2.0        | 0.15.0              | 0.15.0  |
[0.14.0](https://github.com/tensorflow/transform/blob/v0.14.0/RELEASE.md)       | 2.14.0           | 1.14              | 0.14.0              | n/a     |
[0.13.0](https://github.com/tensorflow/transform/blob/v0.13.0/RELEASE.md)       | 2.11.0           | 1.13              | 0.12.1              | n/a     |
[0.12.0](https://github.com/tensorflow/transform/blob/v0.12.0/RELEASE.md)       | 2.10.0           | 1.12              | 0.12.0              | n/a     |
[0.11.0](https://github.com/tensorflow/transform/blob/v0.11.0/RELEASE.md)       | 2.8.0            | 1.11              | 0.9.0               | n/a     |
[0.9.0](https://github.com/tensorflow/transform/blob/v0.9.0/RELEASE.md)         | 2.6.0            | 1.9               | 0.9.0               | n/a     |
[0.8.0](https://github.com/tensorflow/transform/blob/v0.8.0/RELEASE.md)         | 2.5.0            | 1.8               | n/a                 | n/a     |
[0.6.0](https://github.com/tensorflow/transform/blob/v0.6.0/RELEASE.md)         | 2.4.0            | 1.6               | n/a                 | n/a     |
[0.5.0](https://github.com/tensorflow/transform/blob/v0.5.0/RELEASE.md)         | 2.3.0            | 1.5               | n/a                 | n/a     |
[0.4.0](https://github.com/tensorflow/transform/blob/v0.4.0/RELEASE.md)         | 2.2.0            | 1.4               | n/a                 | n/a     |
[0.3.1](https://github.com/tensorflow/transform/blob/v0.3.1/RELEASE.md)         | 2.1.1            | 1.3               | n/a                 | n/a     |
[0.3.0](https://github.com/tensorflow/transform/blob/v0.3.0/RELEASE.md)         | 2.1.1            | 1.3               | n/a                 | n/a     |
[0.1.10](https://github.com/tensorflow/transform/blob/v0.1.10/RELEASE.md)       | 2.0.0            | 1.0               | n/a                 | n/a     |

## Questions

Please direct any questions about working with `tf.Transform` to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-transform](https://stackoverflow.com/questions/tagged/tensorflow-transform)
tag.
