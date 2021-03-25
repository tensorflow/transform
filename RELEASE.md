# Version 0.29.0

## Major Features and Improvements

*   `tft.AnalyzeAndTransformDataset` and `tft.TransformDataset` can now output
    `pyarrow.RecordBatch`es. This is controlled by a parameter
    `output_record_batches` which is set to `False` by default.

## Bug Fixes and Other Changes

*   Added `tft.make_and_track_object` to load and track `tf.Trackable` objects
    created inside the `preprocessing_fn` (for example, tf.hub models). This API
    should only be used when `force_tf_compat_v1=False` and TF2 behavior is
    enabled.
*   The `decode` method of the available coders (`tft.coders.CsvCoder` and
    `tft.coders.ExampleProtoCoder`) have been removed. These were deprecated in
    the 0.25 release.
    [Canned TFXIO implementations](https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/tfxio)
    should be used to read and decode data instead.
*   Previously deprecated APIs were removed: `tft.uniques` (replaced by
    `tft.vocabulary`), `tft.string_to_int` (replaced by
    `tft.compute_and_apply_vocabulary`), `tft.apply_vocab` (replaced by
    `tft.apply_vocabulary`), and `tft.apply_function` (identity function).
*   Removed the `always_return_num_quantiles` arg of `tft.quantiles` and
    `tft.bucketize` which was deprecated in version 0.26.
*   Added support for `count_params` method to the `TransformFeaturesLayer`.
    This will allow to call Keras Model's `summary()` method if the model is
    using the `TransformFeaturesLayer`.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `tensorflow-metadata>=0.29.0,<0.30.0`.
*   Depends on `tfx-bsl>=0.29.0,<0.30.0`.

## Breaking Changes

*   Existing caches (for all analyzers) are automatically invalidated.

## Deprecations

*   N/A
