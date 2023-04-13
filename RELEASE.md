# Version 1.13.0

## Major Features and Improvements

*   `RaggedTensor`s can now be automatically inferred for variable length
    features by setting `represent_variable_length_as_ragged=true` in TFMD
    schema.
*   New experimental APIs added for annotating sparse output tensors:
    `tft.experimental.annotate_sparse_output_shape` and
    `tft.experimental.annotate_true_sparse_output`.
*   `DatasetKey.non_cacheable` added to allow for some datasets to not produce
    cache. This may be useful for gradual cache generation when operating on a
    large rolling range of datasets.
*   Vocabularies produced by `compute_and_apply_vocabulary` can now store
    frequencies. Controlled by the `store_frequency` parameter.

## Bug Fixes and Other Changes

*   Depends on `numpy~=1.22.0`.
*   Depends on `tensorflow>=2.12.0,<2.13`.
*   Depends on `protobuf>=3.20.3,<5`.
*   Depends on `tensorflow-metadata>=1.13.1,<1.14.0`.
*   Depends on `tfx-bsl>=1.13.0,<1.14.0`.
*   Modifies `get_vocabulary_size_by_name` to return a minimum of 1.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.7 support.

