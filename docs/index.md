{% setvar github_path %}tensorflow/transform{% endsetvar %}
{% include "_templates/github-bug.html" %}

# TensorFlow Transform

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

### Dependencies

`tf.Transform` requires TensorFlow but does not depend on the `tensorflow`
[PyPI package](https://pypi.org/project/tensorflow/). See the
[TensorFlow install guides](https://www.tensorflow.org/install/) for
instructions.

[Apache Beam](https://beam.apache.org/) is required to run distributed analysis.
By default, Apache Beam runs in local mode but can also run in distributed mode
using [Google Cloud Dataflow](https://cloud.google.com/dataflow/).
`tf.Transform` is designed to be extensible for other Apache Beam runners.

## Compatible versions

The following table is the `tf.Transform` package versions that are
compatible with each other. This is determined by our testing framework, but
other *untested* combinations may also work.

tensorflow-transform                                                            | tensorflow    | apache-beam[gcp]
------------------------------------------------------------------------------- | ------------- | ----------------
[GitHub master](https://github.com/tensorflow/transform/blob/master/RELEASE.md) | nightly (1.x) | 2.8.0
[0.11.0](https://github.com/tensorflow/transform/blob/v0.11.0/RELEASE.md)       | 1.11          | 2.8.0
[0.9.0](https://github.com/tensorflow/transform/blob/v0.9.0/RELEASE.md)         | 1.9           | 2.6.0
[0.8.0](https://github.com/tensorflow/transform/blob/v0.8.0/RELEASE.md)         | 1.8           | 2.5.0
[0.6.0](https://github.com/tensorflow/transform/blob/v0.6.0/RELEASE.md)         | 1.6           | 2.4.0
[0.5.0](https://github.com/tensorflow/transform/blob/v0.5.0/RELEASE.md)         | 1.5           | 2.3.0
[0.4.0](https://github.com/tensorflow/transform/blob/v0.4.0/RELEASE.md)         | 1.4           | 2.2.0
[0.3.1](https://github.com/tensorflow/transform/blob/v0.3.1/RELEASE.md)         | 1.3           | 2.1.1
[0.3.0](https://github.com/tensorflow/transform/blob/v0.3.0/RELEASE.md)         | 1.3           | 2.1.1
[0.1.10](https://github.com/tensorflow/transform/blob/v0.1.10/RELEASE.md)       | 1.0           | 2.0.0

## Questions

Please direct any questions about working with `tf.Transform` to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-transform](https://stackoverflow.com/questions/tagged/tensorflow-transform)
tag.
