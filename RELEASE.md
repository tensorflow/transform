# Version 0.28.0

## Major Features and Improvements

*   Large vocabularies are now computed faster due to partially parallelizing
    `VocabularyOrderAndWrite`.

## Bug Fixes and Other Changes

*   Generic `tf.SparseTensor` input support has been added to
    `tft.scale_to_0_1`, `tft.scale_to_z_score`, `tft.scale_by_min_max`,
    `tft.min`, `tft.max`, `tft.mean`, `tft.var`, `tft.sum`, `tft.size` and
    `tft.word_count`.
*   Optimize SavedModel written out by `tf.Transform` when using native TF2 to
    speed up loading it.
*   Added `tft_beam.PTransformAnalyzer` as a base PTransform class for
    `tft.ptransform_analyzer` users who wish to have access to a base temporary
    directory.
*   Fix an issue where >2D `SparseTensor`s may be incorrectly represented in
    instance_dicts format.
*   Added support for out-of-vocabulary keys for per_key mappers.
*   Added `tft.get_num_buckets_for_transformed_feature` which provides the
    number of buckets for a transformed feature if it is a direct output of
    `tft.bucketize`, `tft.apply_buckets`, `tft.compute_and_apply_vocabulary` or
    `tft.apply_vocabulary`.
*   Depends on `apache-beam[gcp]>=2.28,<3`.
*   Depends on `numpy>=1.16,<1.20`.
*   Depends on `tensorflow-metadata>=0.28.0,<0.29.0`.
*   Depends on `tfx-bsl>=0.28.1,<0.29.0`.

## Breaking changes

*   Autograph is disabled when the preprocessing fn is traced using tf.function
    when `force_tf_compat_v1=False` and TF2 behavior is enabled.

## Deprecations

*   N/A
