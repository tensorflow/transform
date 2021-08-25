# Version 1.3.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   `tft.quantiles`, `tft.mean` and `tft.var` now ignore NaNs and infinite input
    values. Previously, these would lead to incorrect output calculation.
*   Improved error message for `tft_beam.AnalyzeDataset`,
    `tft_beam.AnalyzeAndTransformDataset` and `tft_beam.AnalyzeDatasetWithCache`
    when the input metadata is empty.
*   Added best-effort TensorFlow Decision Forests (TF-DF) and Struct2Tensor op
    registration when loading transformation graphs.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<2.7`.
*   Depends on `tfx-bsl>=1.3.0,<1.4.0`.

## Breaking Changes

*   Existing `tft.mean` and `tft.var` caches are automatically invalidated.

## Deprecations

*   N/A
