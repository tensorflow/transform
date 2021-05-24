# Version 1.0.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<2.6`.
*   Depends on `tensorflow-metadata>=1.0.0,<1.1.0`.
*   Depends on `tfx-bsl>=1.0.0,<1.1.0`.

## Breaking Changes

*   `tft.ptransform_analyzer` has been moved under `tft.experimental`. The order
    of args in the API has also been changed.
*   `tft_beam.PTransformAnalyzer` has been moved under `tft_beam.experimental`.
*   The default value of the `drop_unused_features` parameter to
   `TFTransformOutput.transform_raw_features` is now True.

## Deprecations

*   N/A
