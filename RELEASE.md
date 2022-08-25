# Version 1.10.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Assign different close_to_resources resource hints to both original and
    cloned PTransforms in deep copy optimization. The reason of adding these
    resource hints is to prevent root Reads that are generated from deep copy
    being merged due to common subexpression elimination.
*   Depends on `apache-beam[gcp]>=2.40,<3`.
*   Depends on `pyarrow>=6,<7`.
*   Depends on `tensorflow-metadata>=1.10.0,<1.11.0`.
*   Depends on `tfx-bsl>=1.10.0,<1.11.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

