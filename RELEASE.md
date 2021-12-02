# Version 1.5.0

## Major Features and Improvements

*   Introduced `tft.experimental.approximate_vocabulary` analyzer that is an
    approximate version of `tft.vocabulary` which is more efficient with smaller
    number of unique elements or `top_k` threshold.

## Bug Fixes and Other Changes

*   Raise a RuntimeError if order of analyzers in traced Tensorflow Graph is
    non-deterministic in TF2.
*   Fix issue where a `tft.experimental.ptransform_analyzer`'s output dtype
    could be propagated incorrectly if it was a primitive as opposed to
    `np.ndarray`.
*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<2.8`.
*   Depends on `tensorflow-metadata>=1.5.0,<1.6.0`.
*   Depends on `tfx-bsl>=1.5.0,<1.6.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

