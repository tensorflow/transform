# Version 1.14.0

## Major Features and Improvements

*   Adds a `reserved_tokens` parameter to vocabulary APIs, a list of tokens that
    must appear in the vocabulary and maintain their order at the beginning of
    the vocabulary.

## Bug Fixes and Other Changes

*   `approximate_vocabulary` now returns tokens with the same frequency in
    reverse lexicographical order (similarly to `tft.vocabulary`).
*   Transformed data batches are now sliced into smaller chunks if their size
    exceeds 200MB.
*   Depends on `pyarrow>=10,<11`.
*   Depends on `apache-beam>=2.47,<3`.
*   Depends on `numpy>=1.22.0`.
*   Depends on `tensorflow>=2.13.0,<3`.

## Breaking Changes

*   Vocabulary related APIs now require passing non-positional parameters by
    key.

## Deprecations

*   N/A

