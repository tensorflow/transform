# Version 0.23.0

## Major Features and Improvements

*   Added EstimatedProbabilityDensityColumn feature column.
*   Added `tft.scale_to_gaussian` to transform input to standard gaussian.
*   Vocabulary related analyzers and mappers now accept a `file_format` argument
    allowing the vocabulary to be saved in TFRecord format. The default format
    remains text (TFRecord format requires tensorflow>=2.4).

## Bug Fixes and Other Changes

*   Enable `SavedModelLoader` to import and apply TF2 SavedModels.
*   `tft.min`, `tft.max`, `tft.sum`, `tft.covariance` and `tft.pca` now have
    default output values to properly process empty analysis datasets.
*   `tft.scale_by_min_max`, `tft.scale_to_0_1` and the corresponding per-key
    versions now apply a sigmoid function to scale tensors if the analysis
    dataset is either empty or contains a single distinct value.
*   Added best-effort tf.text op registration when loading transformation
    graphs.
*   Vocabularies computed over numerical features will now assign values to
    entries with equal frequency in reverse lexicographical order as well,
    similarly to string features.
*   Fixed an issue that causes the `TABLE_INITIALIZERS` graph collection to
    contain a tensor instead of an op when a TF2 SavedModel or a TF2 Hub Module
    containing a table is loaded inside the `preprocessing_fn`.
*   Fixes an issue where the output tensors of `tft.TransformFeaturesLayer`
    would all have unknown shapes.
*   Stopped depending on `avro-python3`.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<2.4`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.

## Breaking changes

*   Existing caches (for all analyzers) are automatically invalidated.

## Deprecations

*   Deprecating Py2 support.
*   Note: We plan to remove Python 3.5 support after this release.
