# Version 0.27.0

## Major Features and Improvements

*   Added `QuantilesCombiner.compact` method that moves some amount of work done
    by `tft.quantiles` from non-parallelizable to parallelizable stage of the
    computation.

## Bug Fixes and Other Changes

*   Strip only newlines instead of all whitespace in the TFTransformOutput
    vocabulary_by_name method.
*   Switch analyzers that output asset files to return an eager tensor
    containing the asset file path instead of a tf.saved_model.Asset object when
    `force_tf_compat_v1=False`. If this file is then used to initialize a table,
    this ensures the input to the `tf.lookup.TextFileInitializer` is the file
    path as the initializer handles wrapping this in a `tf.saved_model.Asset`
    object.
*   Added `tft.annotate_asset` for annotating asset files with a string key that
    can be used to retrieve them in `tft.TFTransformOutput`.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<2.5`.
*   Depends on `tensorflow-metadata>=0.27.0,<0.28.0`.
*   Depends on `tfx-bsl>=0.27.0,<0.28.0`.

## Breaking changes

*   N/A

## Deprecations

*   Parameter `use_tfxio` in the initializer of `Context` is removed (it was
    deprecated in 0.24.0).
