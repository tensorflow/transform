# Version 1.1.0

## Major Features and Improvements

*   Improved resource usage for `tft.vocabulary` when `top_k` is set by removing
    stages performing repetitive sorting.

## Bug Fixes and Other Changes

*   Support invoking Keras models inside the `preprocessing_fn` using
    `tft.make_and_track_object` when `force_tf_compat_v1=False` with TF2
    behaviors enabled.
*   Fix an issue when computing the metadata for a function with automatic
    control dependencies added where dependencies on inputs which should not be
    evaluated was being retained.
*   Census TFT example: wrapped table initialization with a tf.init_scope() in
    order to avoid reinitializing the table for each batch of data.
*   Stopped depending on `six`.
*   Depends on `protobuf>=3.13,<4`.
*   Depends on `tensorflow-metadata>=1.1.0,<1.2.0`.
*   Depends on `tfx-bsl>=1.1.0,<1.2.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A
