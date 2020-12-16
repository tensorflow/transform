# Version 0.26.0

## Major Features and Improvements

*   Initial support added of >2D `SparseTensor`s as inputs and outputs of the
    `preprocessing_fn`. Note that mappers and analyzers may not support those
    yet, and output >2D `SparseTensor`s will have an unkonwn dense shape.

## Bug Fixes and Other Changes

*   Switched to calling tables and initializers within `tf.init_scope` when the
    `preprocessing_fn` is traced using `tf.function` to avoid re-initializing
    them on every invocation of the traced `tf.function`.
*   Switched to a (notably) faster and more accurate implementation of
    `tft.quantiles` analyzer.
*   Fix an issue where graphs become non-hermetic if a TF2 transform_fn is
    loaded in a TF1 Graph context, by making sure all assets are added to the
    `ASSET_FILEPATHS` collection.
*   Depends on `apache-beam[gcp]>=2.25,!=2.26.*,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<2.4`.
*   Depends on `tensorflow-metadata>=0.26.0,<0.27.0`.
*   Depends on `tfx-bsl>=0.26.0,<0.27.0`.

## Breaking changes

*   Existing `tft.quantiles`, `tft.min` and `tft.max` caches are invalidated.

## Deprecations

*   Parameter `always_return_num_quantiles` of `tft.quantiles` and
    `tft.bucketize` is now deprecated. Both now always generate the requested
    number of buckets. Setting `always_return_num_quantiles` will have no effect
    and it will be removed in the next version.
