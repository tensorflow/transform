# Version 1.21.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Added support for Python 3.12 and 3.13.
*   Pinned Bazel version to 7.7.0.
*   Depends on `tensorflow>=2.21.0,<2.22.0`.
*   Depends on `protobuf>=6.0.0,<7.0.0` for Python 3.11+.
*   Updated `pyarrow` dependency to `>14`.
*   Added workarounds for Apache Beam 2.72.0 (Prism runner) incompatibilities in tests, including soft-asserts for metrics in `tft_unit.py` and bypassing a panic in `deep_copy_test.py`.

## Breaking Changes

*   Dropped support for Python 3.9.

## Deprecations

*   N/A
