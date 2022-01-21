# Version 1.6.0

## Major Features and Improvements

*   Introduced `tft.experimental.get_vocabulary_size_by_name` that can retrieve
    the size of a vocabulary computed using `tft.vocabulary` within the
    `preprocessing_fn`.
*   `tft.experimental.ptransform_analyzer` now supports analyzer cache using the
    newly added `tft.experimental.CacheablePTransformAnalyzer` container.
*   `tft.bucketize_per_key` now supports weights.

## Bug Fixes and Other Changes

*   Depends on `numpy>=1.16,<2`.
*   Depends on `apache-beam[gcp]>=2.35,<3`.
*   Depends on `absl-py>=0.9,<2.0.0`.
*   Depends on `tensorflow-metadata>=1.6.0,<1.7.0`.
*   Depends on `tfx-bsl>=1.6.0,<1.7.0`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

