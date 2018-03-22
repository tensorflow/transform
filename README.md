# TensorFlow Transform [![PyPI](https://img.shields.io/pypi/pyversions/tensorflow-transform.svg?style=plastic)](https://github.com/tensorflow/transform)

**TensorFlow Transform** (**tf.Transform**) is a library for preprocessing
data with [TensorFlow](https://www.tensorflow.org). tf.Transform is useful
for preprocessing that requires a full pass the data, such as:

* normalizing an input value by mean and stdev
* integerizing a vocabulary by looking at all input examples for values
* bucketizing inputs based on the observed data distribution

TensorFlow already supports arbitrary manipulations on a single example or
batch of examples. tf.Transform extends the capabilities to support full
passes over the example data.

The output of tf.Transform is exported as a TensorFlow graph for incorporation
into training and serving. Using the same graph for both training and
serving can prevent training/serving skew, because the same transforms are
performed in both scenarios.

**tf.Transform may introduce backwards incompatible changes before version
1.0**.

## Installation and Dependencies

The easiest and recommended way to install tf.Transform is with the PyPI
package.

`pip install tensorflow-transform`

Currently tf.Transform requires that TensorFlow be installed but does not have
an explicit dependency on TensorFlow as a package. See [TensorFlow
documentation](https://www.tensorflow.org/install/) for more information on
installing TensorFlow.

tf.Transform requires Apache Beam to run distributed analysis. Apache Beam
runs in local mode by default, and can also run in distributed mode
using [Google Cloud Dataflow](https://cloud.google.com/dataflow/).
tf.Transform is designed to be extensible to other Apache Beam runners.

### Compatible Versions

This is a table of versions known to be compatible with each other, based on
our testing framework. Other combinations may also work, but are untested.

|tensorflow-transform                                                            |tensorflow    |apache-beam[gcp]|
|--------------------------------------------------------------------------------|--------------|----------------|
|[GitHub master](https://github.com/tensorflow/transform/blob/master/RELEASE.md) |nightly (1.x) |2.4.0           |
|[0.6.0](https://github.com/tensorflow/transform/blob/v0.6.0/RELEASE.md)         |1.6           |2.4.0           |
|[0.5.0](https://github.com/tensorflow/transform/blob/v0.5.0/RELEASE.md)         |1.5           |2.3.0           |
|[0.4.0](https://github.com/tensorflow/transform/blob/v0.4.0/RELEASE.md)         |1.4           |2.2.0           |
|[0.3.1](https://github.com/tensorflow/transform/blob/v0.3.1/RELEASE.md)         |1.3           |2.1.1           |
|[0.3.0](https://github.com/tensorflow/transform/blob/v0.3.0/RELEASE.md)         |1.3           |2.1.1           |
|[0.1.10](https://github.com/tensorflow/transform/blob/v0.1.10/RELEASE.md)       |1.0           |2.0.0           |

## Getting Started

For instructions on using tf.Transform see the [getting started
guide](./getting_started.md).
