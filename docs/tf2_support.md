# Using tf.Transform with TensorFlow 2.x

Starting with the `0.30` release of `tf.Transform`, the default behavior is to
export a TF 2.x SavedModel unless TF 2.x behaviors are explicitly disabled. This
page provides a guide for using `tf.Transform` to export the transform graph as
a TensorFlow 2.x SavedModel.

## New in tf.Transform with TF 2.x

#### Loading Keras models within the `preprocessing_fn`

Please use the `tft.make_and_track_object` API to load Keras models as shown in
the example below.

```python
def preprocessing_fn(inputs):
  keras_model = tft.make_and_track_object(lambda: tf.keras.models.load_model(...), name='_unique_name')
  ...
  return {'keras_model_output': keras_model(inputs[...])}
```

### Using TF 2.x tf.hub modules

TF 2.x hub modules work in `tf.Transform` only when the `preprocessing_fn` is
traced and exported as a TF 2.x SavedModel (this is the default behavior
starting with `tensorflow_transform 0.30`). Please use the
`tft.make_and_track_object` API to load `tf.hub` modules as shown in the example
below.

```python
def preprocessing_fn(inputs):
  hub_module = tft.make_and_track_object(lambda: hub.load(...))
  ...
  return {'hub_module_output': hub_module(inputs[...])}
```

## Potential migration issues

If migrating an existing `tf.Transform` pipeline from TF 1.x to TF 2.x, the
following issues may be encountered:

### RuntimeError: The order of analyzers in your `preprocessing_fn` appears to be non-deterministic.

In TF 2.x, the `preprocessing_fn` provided by the user is traced several times.
If the order in which TFT analyzers are encountered changes with each trace,
this error will be raised. This can be fixed by removing any non-determinism in
the order in which TFT analyzers are invoked.

### Output of `transform_raw_features` does not contain expected feature.

Example exceptions:

```shell
KeyError: \<feature key>
```

or

```shell
\<feature key> not found in features dictionary.
```

[`TFTransformOutput.transform_raw_features`](https://tensorflow.github.io/transform/api_docs/python/tft-beam/#tensorflow_transform.beam.Context)
ignores the `drop_unused_features` parameter and behaves as if it were True.
Please update any usages of the output dictionary from this API to check if the
key you are attempting to retrieve exists in it.

### tf.estimator.BaselineClassifier sees Table not initialized error.

Example exception:

```shell
tensorflow.python.framework.errors_impl.FailedPreconditionError: Table not initialized.
```

Support for Trainer with Estimator based executor is best-effort. While other
estimators work, we have seen issues with table initialization in the
BaselineClassifier. Please
[disable TF 2.x in `tf.Transform`](#retaining-the-legacy-tftransform-behavior).

## Known issues / Features not yet supported

### Outputting vocabularies in TFRecord format is not yet supported.

`tfrecord_gzip` is not yet supported as a valid value for the `file_format`
parameter in `tft.vocabulary` (and other vocabulary APIs).

## Retaining the legacy tf.Transform behavior

If your `tf.Transform` pipeline should not run with TF 2.x, you can retain the
legacy behavior in one of the following ways:

*   Disable TF2 in `tf.Transform` by calling
    `tf.compat.v1.disable_v2_behavior()`
*   Passing `force_tf_compat_v1=True` to `tft_beam.Context` if using
    `tf.Transform` as a standalone library or to the Transform component in TFX.
