# Version 0.24.0

## Major Features and Improvements

*   Added native TF 2 implementation of Transform's Beam APIs -
    `tft.AnalyzeDataset`, `tft.AnalyzeDatasetWithCache`,
    `tft.AnalyzeAndTransformDataset` and `tft.TransformDataset`. The default
    behavior will continue to use Tensorflow's compat.v1 APIs. This can be
    overriden by setting `tft.Context.force_tf_compat_v1=False`. The default
    behavior for TF 2 users will be switched to the new native implementation in
    a future release.

## Bug Fixes and Other Changes

*   Added a small fanout to analyzers' `CombineGlobally` for improved
    performance.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `protobuf>=3.9.2,<4`.
*   Depends on `tensorflow-metadata>=0.24,<0.25`.
*   Depends on `tfx-bsl>=0.24,<0.25`.

## Breaking changes

*   N/A

## Deprecations

*   Deprecating Py3.5 support.
*   Parameter `use_tfxio` in the initializer of `Context` is deprecated. TFT
    Beam APIs now accepts both "instance dicts" and "TFXIO" input formats.
    Setting it will have no effect and it will be removed in the next version.
