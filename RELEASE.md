# Version 1.7.0

## Major Features and Improvements

*   Introduced `tft.experimental.compute_and_apply_approximate_vocabulary` which
    computes and applies an approximate vocabulary.

## Bug Fixes and Other Changes

*   Fix an issue when `tft.experimental.approximate_vocabulary` with `text`
    output format would not filter out tokens with newline characters.
*   Add a dummy value to the result of `tft.experimental.approximate_vocabulary`
    as is done for the exact variant, in order for downstream code to easily
    handle it.
*   Update `tft.get_analyze_input_columns` to ensure its output includes
    `preprocessing_fn` inputs which are not used in any TFT analyzers, but end
    up in a control dependency (automatic control dependencies are not present
    in TF1, hence this change will only affect the native TF2 implementation).
*   Assign different resource hint tags to both orginal and cloned PTransforms
    in deep copy optimization. The reason of adding these tags is to prevent
    root Reads that are generated from deep copy being merged due to common
    subexpression elimination.
*   Fixed an issue when large int64 values would be incorrectly bucketized in
    `tft.apply_buckets`.
*   Depends on `apache-beam[gcp]>=2.36,<3`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.
*   Depends on `tfx-bsl>=1.7.0,<1.8.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

