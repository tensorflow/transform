# Version 1.4.0

## Major Features and Improvements

*   Added `tf.RaggedTensor` support to all analyzers and mappers with
    `reduce_instance_dims=True`.

## Bug Fixes and Other Changes

*   Fix re-loading a transform graph containing pyfuncs exported as a TF1
    SavedModel(added using `tft.apply_pyfunc`) in TF2.
*   Depends on `pyarrow>=1,<6`.
*   Depends on `tensorflow-metadata>=1.4.0,<1.5.0`.
*   Depends on `tfx-bsl>=1.4.0,<1.5.0`.
*   Depends on `apache-beam[gcp]>=2.33,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

