# Version 0.30.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Removed the `dataset_schema` module, most methods in it have been deprecated
    since version 0.14.
*   Fix a bug where having an analyzer operate on the output of `tft.vocabulary`
    would cause it to evaluate incorrectly when `force_tf_compat_v1=False` with
    TF2 behaviors enabled.
*   Depends on `tensorflow-metadata>=0.30.0,<0.31.0`.
*   Depends on `tfx-bsl>=0.30.0,<0.31.0`.

## Breaking Changes

*   `DatasetMetadata` no longer accepts a dict as its input schema. `schema` is
    expected to be a `Schema` proto now.
*   TF 1.15 specific APIs `apply_saved_model` and
    `apply_function_with_checkpoint` were removed from the `tft` namespace. They
    are still available under the `pretrained_models` module.
*   `tft.AnalyzeDataset`, `tft.AnalyzeDatasetWithCache`,
    `tft.AnalyzeAndTransformDataset` and `tft.TransformDataset` will use the
    native TF2 implementation of tf.transform unless TF2 behaviors are
    explicitly disabled. The previous behaviour can still be obtained by setting
    `tft.Context.force_tf_compat_v1=True`.

## Deprecations

*   N/A
