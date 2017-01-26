# tf.Transform

**tf.Transform** is a library for doing data preprocessing with TensorFlow. It
allows users to combine various data processing frameworks (currently Apache
Beam is supported but tf.Transform can be extended to support other frameworks),
with TensorFlow, to transform data. Because tf.Transform is built on TensorFlow,
it allows users to export a graph which re-creates the transformations they did
to their data as a TensorFlow graph. This is important as the user can then
incorporate the exported TensorFlow graph into their serving model, thus
avoiding skew between the served model and the training data.

## tf.Transform Concepts

The most important concept of tf.Transform is the "preprocessing function". This
is a logical description of a transformation of a dataset. The dataset is
conceptualized as a dictionary of columns, and the preprocessing function is
defined by means of two kinds of function:

1) A "transform" which is a function defined using TensorFlow that accepts and
returns tensors. Such a function is applied to some input columns and generates
transformed columns. Users define their own transforms by first defining a
function that operates on tensors, and then applying this to columns using the
`tf_transform.transform` function.

2) An "analyzer" which is a function that accepts columns and returns a
"statistic". A statistic is like a column except that it only has a single
value. An example of an analyzer is `tf_transform.min` which computes the
minimum of a column. Currently tf.Transform provides a fixed set of analyzers.

By combining analyzers and transforms, users can create arbitrary pipelines for
transforming their data. In particular, users should define a "preprocessing
function" which accepts and returns columns.

Columns are not themselves wrappers around data, rather they are placeholders
used to construct a definition of the user's logical pipeline. In order to apply
such a pipeline to data, we rely on the implementation. The Apache Beam
implementation provides `PTransform`s that apply a user's preprocessing function
to data. The typical workflow of a tf.Transform user will be to construct a
preprocessing function, and then incorporate this into a large Beam pipeline,
ultimately materializing the data for training.

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

The easiest way to install tf.Transform is with the PyPI package.

`pip install tensorflow_transform`

Currently tf.Transform requires that TensorFlow be installed but does not have
an explicit dependency on TensorFlow as a package. See [TensorFlow
documentation](https://www.tensorflow.org/get_started/os_setup) for more
information on installing TensorFlow.

This package depends on the Google Cloud Dataflow distribution of Apache Beam.
Apache Beam is the package used to run distributed pipelines. Apache Beam is
able to run pipelines in multiple ways, depending on the "runner" used. While
Apache Beam is an open source package, currently the only distribution on PyPI
is the Cloud Dataflow distribution. This package can run beam pipelines locally,
or on Google Cloud Dataflow.

When a base package for Apache Beam (containing no runners) is available, the
tf.Transform package will depend only on this base package, and users will be
able to install their own runners. tf.Transform will attempt to be as
independent from the specific runner as possible.

## Getting Started

For instructions on using tf.Transform see the [getting started
guide](./getting_started.md)
