# tf.Transform

**tf.Transform** is a library for doing data preprocessing with TensorFlow. It
allows users to combine various data processing frameworks (currently Apache
Beam is supported but tf.Transform can be extended to support other frameworks),
with TensorFlow, to transform data. Because tf.Transform is built on TensorFlow,
it allows users to export a graph which re-creates the transformations they did
to their data as a TensorFlow graph. This is important as the user can then
incorporate the exported TensorFlow graph into their serving model, thus
avoiding skew between the served model and the training data.

**tf.Transform may introduce backwards incompatible changes before version
1.0**.

## Background

While TensorFlow allows users to do arbitrary manipulations on a single instance
or batch of instances, some kinds of preprocessing require a full pass over the
dataset. For example, normalizing an input value, computing a vocabulary for a
string input (and then mapping the string to an int with this vocabulary), or
bucketizing an input. While some of these operations can be done with TensorFlow
in a streaming manner (e.g. calculating a running mean for normalization), in
general it may be preferable or necessary to calculate these with a full pass
over the data.

## Installation and Dependencies

The easiest and recommended way to install tf.Transform is with the PyPI
package.

`pip install tensorflow-transform`

Currently tf.Transform requires that TensorFlow be installed but does not have
an explicit dependency on TensorFlow as a package. See [TensorFlow
documentation](https://www.tensorflow.org/install/) for more information on
installing TensorFlow.

tf.Transform does though have a dependency on the Google Cloud Dataflow
distribution of Apache Beam.
Apache Beam is the package used to run distributed pipelines. Apache Beam is
able to run pipelines in multiple ways, depending on the "runner" used. While
Apache Beam is an open source package, currently the only distribution on PyPI
is the Cloud Dataflow distribution. This package can run beam pipelines locally,
or on Google Cloud Dataflow.

When a base package for Apache Beam (containing no runners) is available, the
tf.Transform package will depend only on this base package, and users will be
able to install their own runners. tf.Transform will attempt to be as
independent from the specific runner as possible.

Note: If you clone tf.Transform's implementation and samples from GitHub's
`master` branch (as opposed to using the released implementation and samples
from PyPI) they will likely only work with TensorFlow's nightly
[build](https://github.com/tensorflow/tensorflow).

### Compatible Versions

This is a table of versions known to be compatible with each other.  This is not
a comprehensive list, meaning other combinations may also work, but these are
the combinations tested by our testing framework and by the team before
releasing a new version.

|tensorflow-transform                                                            |tensorflow|apache-beam[gcp]|
|--------------------------------------------------------------------------------|----------|----------------|
|[GitHub master](https://github.com/tensorflow/transform/blob/master/RELEASE.md) |nightly   |latest (2.x)    |
|[0.3.0](https://github.com/tensorflow/transform/blob/v0.3.0/RELEASE.md)         |1.3       |2.1.1           |
|[0.1.10](https://github.com/tensorflow/transform/blob/v0.1.10/RELEASE.md)       |1.0       |2.0.0           |
## Getting Started

For instructions on using tf.Transform see the [getting started
guide](./getting_started.md).
