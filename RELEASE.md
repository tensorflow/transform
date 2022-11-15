# Version 1.11.0

## Major Features and Improvements

*   This is the last version that supports TensorFlow 1.15.x. TF 1.15.x support
    will be removed in the next version. Please check the
    [TF2 migration guide](https://www.tensorflow.org/guide/migrate) to migrate
    to TF2.

*   Introduced `tft.experimental.document_frequency` and `tft.experimental.idf`
    which map each term to its document frequency and inverse document frequency
    in the same order as the terms in documents.
*   `schema_utils.schema_as_feature_spec` now supports struct features as a way
    to describe `tf.SequenceExample` data.
*   TensorRepresentations in schema used for
    `schema_utils.schema_as_feature_spec` can now share name with their source
    features.
*   Introduced `tft_beam.EncodeTransformedDataset` which can be used to easily
    encode transformed data in preparation for materialization.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.10,<2.11`
*   Depends on `apache-beam[gcp]>=2.41,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

