<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug Fixes and Other Changes

## Breaking Changes

## Deprecations

# Version 1.17.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow 2.17`
*   Depends on `protobuf>=4.25.2,<6.0.0` for Python 3.11 and on `protobuf>4.21.6,<6.0.0` for 3.9 and 3.10.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on
    `apache-beam[gcp]>=2.50.0,<2.51.0` for 3.9 and 3.10.
*   macOS wheel publishing is temporarily paused due to missing ARM64 support.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.16.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow 2.16`
*   Relax dependency on Protobuf to include version 5.x

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.15.0

## Major Features and Improvements

*   Added support for sparse labels in AMI vocabulary computation.

## Bug Fixes and Other Changes

*   Bumped the Ubuntu version on which `tensorflow_transform` is tested to 20.04
    (previously was 16.04).
*   Explicitly use Keras 2 or `tf_keras`` if Keras 3 is installed.
*   Added python 3.11 support.
*   Depends on `tensorflow 2.15`.
*   Enable passing `tf.saved_model.SaveOptions` to model saving functionality.
*   Census and sentiment examples updated to only use Keras instead of
    estimator.
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on 
    `apache-beam[gcp]>=2.47.0,<3` for 3.9 and 3.10.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.

## Breaking Changes

*   Existing analyzer cache is automatically invalidated.

## Deprecations

*   Deprecated python 3.8 support.

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

# Version 1.12.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.11,<2.12`
*   Depends on `tensorflow-metadata>=1.12.0,<1.13.0`.
*   Depends on `tfx-bsl>=1.12.0,<1.13.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

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

# Version 1.9.0

## Major Features and Improvements

*   Adds element-wise scaling support to `scale_by_min_max_per_key`,
    `scale_to_0_1_per_key` and `scale_to_z_score_per_key` for
    `key_vocabulary_filename = None`.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.9,<2.10`
*   Depends on `tensorflow-metadata>=1.9.0,<1.10.0`.
*   Depends on `tfx-bsl>=1.9.0,<1.10.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.8.0

## Major Features and Improvements

*   Adds `tft.DatasetMetadata` and its factory method `from_feature_spec` as
    public APIs to be used when using the "instance dict" data format.

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.38,<3`.
*   Depends on `tensorflow-metadata>=1.8.0,<1.9.0`.
*   Depends on `tfx-bsl>=1.8.0,<1.9.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

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
*   Assign different resource hint tags to both original and cloned PTransforms
    in deep copy optimization. The reason of adding these tags is to prevent
    root Reads that are generated from deep copy being merged due to common
    subexpression elimination.
*   Fixed an issue when large int64 values would be incorrectly bucketized in
    `tft.apply_buckets`.
*   Depends on `apache-beam[gcp]>=2.36,<3`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<2.9`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.
*   Depends on `tfx-bsl>=1.7.0,<1.8.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.6.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<2.9`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.6.0

## Major Features and Improvements

*   Introduced `tft.experimental.get_vocabulary_size_by_name` that can retrieve
    the size of a vocabulary computed using `tft.vocabulary` within the
    `preprocessing_fn`.
*   `tft.experimental.ptransform_analyzer` now supports analyzer cache using the
    newly added `tft.experimental.CacheablePTransformAnalyzer` container.
*   `tft.bucketize_per_key` now supports weights.

## Bug Fixes and Other Changes

*   Depends on `numpy>=1.16,<2`.
*   Depends on `apache-beam[gcp]>=2.35,<3`.
*   Depends on `absl-py>=0.9,<2.0.0`.
*   Depends on `tensorflow-metadata>=1.6.0,<1.7.0`.
*   Depends on `tfx-bsl>=1.6.0,<1.7.0`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<2.8`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.5.0

## Major Features and Improvements

*   Introduced `tft.experimental.approximate_vocabulary` analyzer that is an
    approximate version of `tft.vocabulary` which is more efficient with smaller
    number of unique elements or `top_k` threshold.

## Bug Fixes and Other Changes

*   Raise a RuntimeError if order of analyzers in traced Tensorflow Graph is
    non-deterministic in TF2.
*   Fix issue where a `tft.experimental.ptransform_analyzer`'s output dtype
    could be propagated incorrectly if it was a primitive as opposed to
    `np.ndarray`.
*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<2.8`.
*   Depends on `tensorflow-metadata>=1.5.0,<1.6.0`.
*   Depends on `tfx-bsl>=1.5.0,<1.6.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.4.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `future` package.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.4.0

## Major Features and Improvements

*   Added `tf.RaggedTensor` support to all analyzers and mappers with
    `reduce_instance_dims=True`.

## Bug Fixes and Other Changes

*   Fix re-loading a transform graph containing pyfuncs exported as a TF1
    SavedModel(added using `tft.apply_pyfunc`) in TF2.
*   Depends on `pyarrow>=1,<6`.
*   Depends on `tensorflow-metadata>=1.4.0,<1.5.0`.
*   Depends on `tfx-bsl>=1.4.0,<1.5.0`.
*   Depends on `apache-beam[gcp]>=2.33,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

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

# Version 1.2.0

## Major Features and Improvements

*   Added `RaggedTensor` support to output schema inference and transformed
    tensors conversion to instance dicts and `pa.RecordBatch` with TF 2.x.

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `tensorflow-metadata>=1.2.0,<1.3.0`.
*   Depends on `tfx-bsl>=1.2.0,<1.3.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.1.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `google-cloud-bigquery>>=1.28.0,<2.21`.
*   Depends on `tfx-bsl>=1.1.0,<1.2.0`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

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

# Version 0.30.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Removed the `dataset_schema` module, most methods in it have been deprecated
    since version 0.14.
*   Fix a bug where having an analyzer operate on the output of `tft.vocabulary`
    would cause it to evaluate incorrectly when `force_tf_compat_v1=False` with
    TF2 behaviors enabled.
*   Depends on `tensorflow-metadata>=0.30.0,<0.31.0`.
*   Depends on `tfx-bsl>=0.30.0,<0.31.0`.

## Breaking Changes

*   `DatasetMetadata` no longer accepts a dict as its input schema. `schema` is
    expected to be a `Schema` proto now.
*   TF 1.15 specific APIs `apply_saved_model` and
    `apply_function_with_checkpoint` were removed from the `tft` namespace. They
    are still available under the `pretrained_models` module.
*   `tft.AnalyzeDataset`, `tft.AnalyzeDatasetWithCache`,
    `tft.AnalyzeAndTransformDataset` and `tft.TransformDataset` will use the
    native TF2 implementation of tf.transform unless TF2 behaviors are
    explicitly disabled. The previous behaviour can still be obtained by setting
    `tft.Context.force_tf_compat_v1=True`.

## Deprecations

*   N/A

# Version 0.29.0

## Major Features and Improvements

*   `tft.AnalyzeAndTransformDataset` and `tft.TransformDataset` can now output
    `pyarrow.RecordBatch`es. This is controlled by a parameter
    `output_record_batches` which is set to `False` by default.

## Bug Fixes and Other Changes

*   Added `tft.make_and_track_object` to load and track `tf.Trackable` objects
    created inside the `preprocessing_fn` (for example, tf.hub models). This API
    should only be used when `force_tf_compat_v1=False` and TF2 behavior is
    enabled.
*   The `decode` method of the available coders (`tft.coders.CsvCoder` and
    `tft.coders.ExampleProtoCoder`) have been removed. These were deprecated in
    the 0.25 release.
    [Canned TFXIO implementations](https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/tfxio)
    should be used to read and decode data instead.
*   Previously deprecated APIs were removed: `tft.uniques` (replaced by
    `tft.vocabulary`), `tft.string_to_int` (replaced by
    `tft.compute_and_apply_vocabulary`), `tft.apply_vocab` (replaced by
    `tft.apply_vocabulary`), and `tft.apply_function` (identity function).
*   Removed the `always_return_num_quantiles` arg of `tft.quantiles` and
    `tft.bucketize` which was deprecated in version 0.26.
*   Added support for `count_params` method to the `TransformFeaturesLayer`.
    This will allow to call Keras Model's `summary()` method if the model is
    using the `TransformFeaturesLayer`.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `tensorflow-metadata>=0.29.0,<0.30.0`.
*   Depends on `tfx-bsl>=0.29.0,<0.30.0`.

## Breaking Changes

*   Existing caches (for all analyzers) are automatically invalidated.

## Deprecations

*   N/A

# Version 0.28.0

## Major Features and Improvements

*   Large vocabularies are now computed faster due to partially parallelizing
    `VocabularyOrderAndWrite`.

## Bug Fixes and Other Changes

*   Generic `tf.SparseTensor` input support has been added to
    `tft.scale_to_0_1`, `tft.scale_to_z_score`, `tft.scale_by_min_max`,
    `tft.min`, `tft.max`, `tft.mean`, `tft.var`, `tft.sum`, `tft.size` and
    `tft.word_count`.
*   Optimize SavedModel written out by `tf.Transform` when using native TF2 to
    speed up loading it.
*   Added `tft_beam.PTransformAnalyzer` as a base PTransform class for
    `tft.ptransform_analyzer` users who wish to have access to a base temporary
    directory.
*   Fix an issue where >2D `SparseTensor`s may be incorrectly represented in
    instance_dicts format.
*   Added support for out-of-vocabulary keys for per_key mappers.
*   Added `tft.get_num_buckets_for_transformed_feature` which provides the
    number of buckets for a transformed feature if it is a direct output of
    `tft.bucketize`, `tft.apply_buckets`, `tft.compute_and_apply_vocabulary` or
    `tft.apply_vocabulary`.
*   Depends on `apache-beam[gcp]>=2.28,<3`.
*   Depends on `numpy>=1.16,<1.20`.
*   Depends on `tensorflow-metadata>=0.28.0,<0.29.0`.
*   Depends on `tfx-bsl>=0.28.1,<0.29.0`.

## Breaking changes

*   Autograph is disabled when the preprocessing fn is traced using tf.function
    when `force_tf_compat_v1=False` and TF2 behavior is enabled.

## Deprecations

# Version 0.27.0

## Major Features and Improvements

*   Added `QuantilesCombiner.compact` method that moves some amount of work done
    by `tft.quantiles` from non-parallelizable to parallelizable stage of the
    computation.

## Bug Fixes and Other Changes

*   Strip only newlines instead of all whitespace in the TFTransformOutput
    vocabulary_by_name method.
*   Switch analyzers that output asset files to return an eager tensor
    containing the asset file path instead of a tf.saved_model.Asset object when
    `force_tf_compat_v1=False`. If this file is then used to initialize a table,
    this ensures the input to the `tf.lookup.TextFileInitializer` is the file
    path as the initializer handles wrapping this in a `tf.saved_model.Asset`
    object.
*   Added `tft.annotate_asset` for annotating asset files with a string key that
    can be used to retrieve them in `tft.TFTransformOutput`.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<2.5`.
*   Depends on `tensorflow-metadata>=0.27.0,<0.28.0`.
*   Depends on `tfx-bsl>=0.27.0,<0.28.0`.

## Breaking changes

*   N/A

## Deprecations

*   Parameter `use_tfxio` in the initializer of `Context` is removed (it was
    deprecated in 0.24.0).

# Version 0.26.0

## Major Features and Improvements

*   Initial support added of >2D `SparseTensor`s as inputs and outputs of the
    `preprocessing_fn`. Note that mappers and analyzers may not support those
    yet, and output >2D `SparseTensor`s will have an unknown dense shape.

## Bug Fixes and Other Changes

*   Switched to calling tables and initializers within `tf.init_scope` when the
    `preprocessing_fn` is traced using `tf.function` to avoid re-initializing
    them on every invocation of the traced `tf.function`.
*   Switched to a (notably) faster and more accurate implementation of
    `tft.quantiles` analyzer.
*   Fix an issue where graphs become non-hermetic if a TF2 transform_fn is
    loaded in a TF1 Graph context, by making sure all assets are added to the
    `ASSET_FILEPATHS` collection.
*   Depends on `apache-beam[gcp]>=2.25,!=2.26.*,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<2.4`.
*   Depends on `tensorflow-metadata>=0.26.0,<0.27.0`.
*   Depends on `tfx-bsl>=0.26.0,<0.27.0`.

## Breaking changes

*   Existing `tft.quantiles`, `tft.min` and `tft.max` caches are invalidated.

## Deprecations

*   Parameter `always_return_num_quantiles` of `tft.quantiles` and
    `tft.bucketize` is now deprecated. Both now always generate the requested
    number of buckets. Setting `always_return_num_quantiles` will have no effect
    and it will be removed in the next version.

# Version 0.25.0

## Major Features and Improvements

*   Updated the "Getting Started" guide and examples to demonstrate the support
    for both the "instance dict" and the "TFXIO" format. Users are encouraged to
    start using the "TFXIO" format, expecially in cases where
    [pre-canned TFXIO implementations](https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/tfxio)
    is available as it offers better performance.
*   From this release TFT will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple tensorflow-transform
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFT available on PyPI by running the
    command `pip install tensorflow-transform` .

## Bug Fixes and Other Changes

*   `TFTransformOutput.transform_raw_features` and `TransformFeaturesLayer` can
    be used when a transform fn is exported as a TF2 SavedModel and imported in
    graph mode.
*   Utility methods in `tft.inspect_preprocessing_fn` now take an optional
    parameter `force_tf_compat_v1`. If this is False, the `preprocessing_fn` is
    traced using tf.function in TF 2.x when TF 2 behaviors are enabled.
*   Switching to a wrapper for `collections.namedtuple` to ensure compatibility
    with PySpark which modifies classes produced by the factory.
*   Caching has been disabled for `tft.tukey_h_params`, `tft.tukey_location` and
    `tft.tukey_scale` due to the cached accumulator being non-deterministic.
*   Track variables created within the `preprocessing_fn` in the native TF 2
    implementation.
*   `TFTransformOutput.transform_raw_features` returns a wrapped python dict
    that overrides pop to return None instead of raising a KeyError when called
    with a key not found in the dictionary. This is done as preparation for
    switching the default value of `drop_unused_features` to True.
*   Vocabularies written in `tfrecord_gzip` format no longer filter out entries
    that are empty or that include a newline character.
*   Depends on `apache-beam[gcp]>=2.25,<3`.
*   Depends on `tensorflow-metadata>=0.25,<0.26`.
*   Depends on `tfx-bsl>=0.25,<0.26`.

## Breaking changes

*   N/A

## Deprecations

*   The `decode` method of the available coders (`tft.coders.CsvCoder` and
    `tft.coders.ExampleProtoCoder`) has been deprecated and removed.
    [Canned TFXIO implementations](https://www.tensorflow.org/tfx/tfx_bsl/api_docs/python/tfx_bsl/public/tfxio)
    should be used to read and decode data instead.

# Release 0.24.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.24,<3`.
*   Depends on `tfx-bsl>=0.24.1,<0.25`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.24.0

## Major Features and Improvements

*   Added native TF 2 implementation of Transform's Beam APIs -
    `tft.AnalyzeDataset`, `tft.AnalyzeDatasetWithCache`,
    `tft.AnalyzeAndTransformDataset` and `tft.TransformDataset`. The default
    behavior will continue to use Tensorflow's compat.v1 APIs. This can be
    overridden by setting `tft.Context.force_tf_compat_v1=False`. The default
    behavior for TF 2 users will be switched to the new native implementation in
    a future release.

## Bug Fixes and Other Changes

*   Added a small fanout to analyzers' `CombineGlobally` for improved
    performance.
*   `TransformFeaturesLayer` can be called after being saved as an attribute to
    a Keras Model, even if the layer isn't used in the Model.
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

# Version 0.23.0

## Major Features and Improvements

*   Added `tft.scale_to_gaussian` to transform input to standard gaussian.
*   Vocabulary related analyzers and mappers now accept a `file_format` argument
    allowing the vocabulary to be saved in TFRecord format. The default format
    remains text (TFRecord format requires tensorflow>=2.4).

## Bug Fixes and Other Changes

*   Enable `SavedModelLoader` to import and apply TF2 SavedModels.
*   `tft.min`, `tft.max`, `tft.sum`, `tft.covariance` and `tft.pca` now have
    default output values to properly process empty analysis datasets.
*   `tft.scale_by_min_max`, `tft.scale_to_0_1` and the corresponding per-key
    versions now apply a sigmoid function to scale tensors if the analysis
    dataset is either empty or contains a single distinct value.
*   Added best-effort tf.text op registration when loading transformation
    graphs.
*   Vocabularies computed over numerical features will now assign values to
    entries with equal frequency in reverse lexicographical order as well,
    similarly to string features.
*   Fixed an issue that causes the `TABLE_INITIALIZERS` graph collection to
    contain a tensor instead of an op when a TF2 SavedModel or a TF2 Hub Module
    containing a table is loaded inside the `preprocessing_fn`.
*   Fixes an issue where the output tensors of `tft.TransformFeaturesLayer`
    would all have unknown shapes.
*   Stopped depending on `avro-python3`.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<2.4`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.

## Breaking changes

*   Existing caches (for all analyzers) are automatically invalidated.

## Deprecations

*   Deprecating Py2 support.
*   Note: We plan to remove Python 3.5 support after this release.

# Version 0.22.0

## Major Features and Improvements

## Bug Fixes and Other Changes
* `tft.bucketize_per_key` no longer assumes that the keys during
  transformation existed in the analysis dataset. If a key is missing then the
  assigned bucket will be -1.
* `tft.estimated_probability_density`, when `categorical=True`, no longer
  assumes that the values during transformation existed in the analysis dataset,
  and will assume 0 density in that case.
* Switched analyzer cache representation of dataset keys from using a primitive
  str to a DatasetKey class.
* `tft_beam.analyzer_cache.ReadAnalysisCacheFromFS` can now filter cache entry
  keys when given a `cache_entry_keys` parameter. `cache_entry_keys` can be
  produced by utilizing `get_analysis_cache_entry_keys`.
* Reduced number of shuffles via packing multiple combine merges into a
  single Beam combiner.
* Switch `tft.TransformFeaturesLayer` to use the TF 2 `tf.saved_model.load` API
  to load a previously exported SavedModel.
* Adds `tft.sparse_tensor_left_align` as a utility which aligns
 `tf.SparseTensor`s to the left.
* Depends on `avro-python3>=1.8.1,!=1.9.2.*,<2.0.0` for Python3.5 + MacOS.
* Depends on `apache-beam[gcp]>=2.20.0,<3`.
* Depends on `tensorflow>=1.15,!=2.0.*,<2.3`.
* Depends on `tensorflow-metadata>=0.22.0,<0.23.0`.
* Depends on `tfx-bsl>=0.22.0,<0.23.0`.

## Breaking changes
* `tft.AnalyzeDatasetWithCache` no longer accepts a flat pcollection as an
  input. Instead it will flatten the datasets in the `input_values_pcoll_dict`
  input if needed.
* `tft.TransformFeaturesLayer` no longer takes a parameter
  `drop_unused_features`. Its default behavior is now equivalent to having set
  `drop_unused_features` to `True`.

## Deprecations

# Release 0.21.2

## Major Features and Improvements
* Expanded capability for per-key analyzers to analyze larger sets of keys that
  would not fit in memory, by storing the key-value pairs in vocabulary files.
  This is enabled by passing a `per_key_filename` to `tft.count_per_key` and
  `tft.scale_to_z_score_per_key`.
* Added `tft.TransformFeaturesLayer` and
  `tft.TFTransformOutput.transform_features_layers` to allow transforming
  features for a TensorFlow Keras model.

## Bug Fixes and Other Changes

* `tft.apply_buckets_with_interpolation` now handles NaN values by imputing with
  the middle of the normalized range.
* Depends on `tfx-bsl>=0.21.3,<0.22`.

## Breaking changes

## Deprecations


# Release 0.21.0

## Major Features and Improvements
* Added a new version of the census example to demonstrate usage in TF 2.0.
* New mapper `estimated_probability_density` to compute either exact
  probabilities (for discrete categorical variable) or approximate density over
  fixed intervals (continuous variables).
* New analyzers `count_per_key` and `histogram` to return counts of unique
  elements or values within predefined ranges. Calling `tft.histogram` on
  non-categorical value will assign each data point to the appropriate fixed
  bucket and then count for each bucket.
* Provided capability for per-key analyzers to analyze larger sets of keys that
  would not fit in memory, by storing the key-value pairs in vocabulary files.
  This is enabled by passing a `per_key_filename` to
  `tft.scale_by_min_max_per_key` and `tft.scale_to_0_1_per_key`.

## Bug Fixes and Other Changes
* Added beam counters to log analyzer and mapper usage.
* Cleanup deprecated APIs used in census and sentiment examples.
* Support windows style paths in `analyzer_cache`.
* `tft_beam.WriteTransformFn` and `tft_beam.WriteMetadata` have been made
  idempotent to allow retrying them in case of a failure.
* `tft_beam.WriteMetadata` takes an optional argument `write_to_unique_subdir`
  and returns the path to which metadata was written. If
  `write_to_unique_subdir` is True, metadata is written to a unique subdirectory
  under `path`, otherwise it is written to `path`.
* Support non utf-8 characters when reading vocabularies in
  `tft.TFTransformOutput`
* `tft.TFTransformOutput.vocabulary_by_name` now returns bytes instead of str
  with python 3.

## Breaking changes

## Deprecations

# Release 0.15.0

## Major Features and Improvements
* This release introduces initial beta support for TF 2.0. TF 2.0 programs
  running in "safety" mode (i.e. using TF 1.X APIs through the
  `tensorflow.compat.v1` compatibility module are expected to work. Newly
  written TF 2.0 programs may not work if they exercise functionality that is
  not yet supported. If you do encounter an issue when using
  `tensorflow-transform` with TF 2.0, please create an issue
  https://github.com/tensorflow/transform/issues with instructions on how to
  reproduce it.
* Performance improvements for `preprocessing_fns` with many Quantiles
  analyzers.
* `tft.quantiles` and `tft.bucketize` are now using new TF core quantiles ops
  instead of contrib ops.
* Performance improvements due to packing multiple combine analyzers into a
  single Beam Combiner.

## Bug Fixes and Other Changes
* Existing analyzer cache is invalidated.
* Saved transforms now support composite tensors (such as `tf.RaggedTensor`).
* Vocabulary's cache coder now supports non utf-8 encodable tokens.
* Fixes encoding of the `tft.covariance` accumulator cache.
* Fixes encoding per-key analyzers accumulator cache.
* Make various utility methods in `tft.inspect_preprocessing_fn` support
  `tf.RaggedTensor`.
* Moved beam/shared lib to `tfx-bsl`. If running with latest master, `tfx-bsl`
  must also be latest master.
* `preprocessing_fn`s now have beta support of calls to `tf.function`s, as long
  as they don't contain calls to `tf.Transform` analyzers/mappers or table
  initializers.
* `tft.quantiles` and `tft.bucketize` are now using core TF ops.
* Depends on `tfx-bsl>=0.15,<0.16`.
* Depends on `tensorflow-metadata>=0.15,<0.16`.
* Depends on `apache-beam[gcp]>=2.16,<3`.
* Depends on `tensorflow>=0.15,<2.2`.
  * Starting from 1.15, package
    `tensorflow` comes with GPU support. Users won't need to choose between
    `tensorflow` and `tensorflow-gpu`.
  * Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU
    support. If `tensorflow-gpu` 2.0.0 is installed before installing
    `tensorflow-transform`, it will be replaced with `tensorflow` 2.0.0.
    Re-install `tensorflow-gpu` 2.0.0 if needed.

## Breaking changes
* `always_return_num_quantiles` changed to default to True in `tft.quantiles`
  and `tft.bucketize`, resulting in exact bucket count returned.
* Removes the `input_fn_maker` module which has been deprecated since TFT 0.11.
  For idiomatic construction of `input_fn`, see `tensorflow_transform` examples.

## Deprecations

# Release 0.14.0

## Major Features and Improvements
* New `tft.word_count` mapper to identify the number of tokens for each row
  (for pre-tokenized strings).
* All `tft.scale_to_*` mappers now have per-key variants, along with analyzers
  for `mean_and_var_per_key` and `min_and_max_per_key`.
* New `tft_beam.AnalyzeDatasetWithCache` allows analyzing ranges of data while
  producing and utilizing cache.  `tft.analyzer_cache` can help read and write
  such cache to a filesystem between runs.  This caching feature is worth using
  when analyzing a rolling range in a continuous pipeline manner.  This is an
  experimental feature.
* Added `reduce_instance_dims` support to `tft.quantiles` and `elementwise` to
  `tft.bucketize`, while avoiding separate beam calls for each feature.

## Bug Fixes and Other Changes
* `sparse_tensor_to_dense_with_shape` now accepts an optional `default_value`
  parameter.
* `tft.vocabulary` and `tft.compute_and_apply_vocabulary` now support
  `fingerprint_shuffle` to sort the vocabularies by fingerprint instead of
  counts. This is useful for load balancing the training parameter servers.
  This is an experimental feature.
* Fix numerical instability in `tft.vocabulary` mutual information calculations.
* `tft.vocabulary` and `tft.compute_and_apply_vocabulary` now support computing
  vocabularies over integer categoricals and multivalent input features, and
  computing mutual information for non-binary labels.
* New numeric normalization method available:
  `tft.apply_buckets_with_interpolation`.
* Changes to make this library more compatible with TensorFlow 2.0.
* Fix sanitizing of vocabulary filenames.
* Emit a friendly error message when context isn't set.
* Analyzer output dtypes are enforced to be TensorFlow dtypes, and by extension
  `ptransform_analyzer`'s `output_dtypes` is enforced to be a list of TensorFlow
  dtypes.
* Make `tft.apply_buckets_with_interpolation` support SparseTensors.
* Adds an experimental api for analyzers to annotate the post-transform schema.
* `TFTransformOutput.transform_raw_features` now accepts an optional
  `drop_unused_features` parameter to exclude unused features in output.
* If not specified, the min_diff_from_avg parameter of `tft.vocabulary` now
  defaults to a reasonable value based on the size of the dataset (relevant
  only if computing vocabularies using mutual information).
* Convert some `tf.contrib` functions to be compatible with TF2.0.
* New `tft.bag_of_words` mapper to compute the unique set of ngrams for each row
  (for pre-tokenized strings).
* Fixed a bug in `tf_utils.reduce_batch_count_mean_and_var`, and as a result
  `mean_and_var` analyzer, was miscalculating variance for the sparse
  elementwise=True case.
* At test utility `tft_unit.cross_named_parameters` for creating parameterized
  tests that involve the cartesian product of various parameters.
* Depends on `tensorflow-metadata>=0.14,<0.15`.
* Depends on `apache-beam[gcp]>=2.14,<3`.
* Depends on `numpy>=1.16,<2`.
* Depends on `absl-py>=0.7,<2`.
* Allow `preprocessing_fn` to emit a `tf.RaggedTensor`.  In this case, the
  output `Schema` proto will not be able to be converted to a feature spec,
  and so the output data will not be able to be materialized with `tft.coders`.
* Ability to directly set exact `num_buckets` with new parameter
  `always_return_num_quantiles` for `analyzers.quantiles` and
  `mappers.bucketize`, defaulting to False in general but True when
  `reduce_instance_dims` is False.

## Breaking changes
* `tf_utils.reduce_batch_count_mean_and_var`, which feeds into
  `tft.mean_and_var`, now returns 0 instead of inf for empty columns of a
  sparse tensor.
* `tensorflow_transform.tf_metadata.dataset_schema.Schema` class is removed.
  Wherever a `dataset_schema.Schema` was used, users should now provide a
  `tensorflow_metadata.proto.v0.schema_pb2.Schema` proto. For backwards
  compatibility, `dataset_schema.Schema` is now a factory method that produces
  a `Schema` proto.  Updating code should be straightforward because the
  `dataset_schema.Schema` class was already a wrapper around the `Schema` proto.
* Only explicitly public analyzers are exported to the `tft` module, e.g.
  combiners are no longer exported and have to be accessed directly through
  `tft.analyzers`.
* Requires pre-installed TensorFlow >=1.14,<2.

## Deprecations
* `DatasetSchema` is now a deprecated factory method (see above).
* `tft.tf_metadata.dataset_schema.from_feature_spec` is now deprecated.
  Equivalent functionality is provided by
  `tft.tf_metadata.schema_utils.schema_from_feature_spec`.

# Release 0.13.0

## Major Features and Improvements
* Now `AnalyzeDataset`, `TransformDataset` and `AnalyzeAndTransformDataset` can
  accept input data that only contains columns needed for that operation as
  opposed to all columns defined in schema. Utility methods to infer the list of
  needed columns are added to `tft.inspect_preprocessing_fn`. This makes it
  easier to take advantage of columnar projection when data is stored in
  columnar storage formats.
* Python 3.5 is supported.

## Bug Fixes and Other Changes
* Version is now accessible as `tensorflow_transform.__version__`.
* Depends on `apache-beam[gcp]>=2.11,<3`.
* Depends on `protobuf>=3.7,<4`.

## Breaking changes
* Coders now return index and value features rather than a combined feature for
  `SparseFeature`.
* Requires pre-installed TensorFlow >=1.13,<2.

## Deprecations

# Release 0.12.0

## Major Features and Improvements
* Python 3.5 readiness complete (all tests pass). Full Python 3.5 compatibility
  is expected to be available with the next version of Transform (after
  Apache Beam 2.11 is released).
* Performance improvements for vocabulary generation when using top_k.
* New optimized highly experimental API for analyzing a dataset was added,
  `AnalyzeDatasetWithCache`, which allows reading and writing analyzer cache.
* Update `DatasetMetadata` to be a wrapper around the
  `tensorflow_metadata.proto.v0.schema_pb2.Schema` proto.  TensorFlow Metadata
  will be the schema used to define data parsing across TFX.  The serialized
  `DatasetMetadata` is now the `Schema` proto in ascii format, but the previous
  format can still be read.
* Change `ApplySavedModel` implementation to use `tf.Session.make_callable`
  instead of `tf.Session.run` for improved performance.

## Bug Fixes and Other Changes

* `tft.vocabulary` and `tft.compute_and_apply_vocabulary` now support
  filtering based on adjusted mutual information when
  `use_adjusetd_mutual_info` is set to True.
* `tft.vocabulary` and `tft.compute_and_apply_vocabulary` now takes
  regularization term 'min_diff_from_avg' that adjusts mutual information to
  zero whenever the difference between count of the feature with any label and
  its expected count is lower than the threshold.
* Added an option to `tft.vocabulary` and `tft.compute_and_apply_vocabulary`
  to compute a coverage vocabulary, using the new `coverage_top_k`,
  `coverage_frequency_threshold` and `key_fn` parameters.
* Added `tft.ptransform_analyzer` for advanced use cases.
* Modified `QuantilesCombiner` to use `tf.Session.make_callable` instead of
  `tf.Session.run` for improved performance.
* ExampleProtoCoder now also supports non-serialized Example representations.
* `tft.tfidf` now accepts a scalar Tensor as `vocab_size`.
* `assertItemsEqual` in unit tests are replaced by `assertCountEqual`.
* `NumPyCombiner` now outputs TF dtypes in output_tensor_infos instead of
  numpy dtypes.
* Adds function `tft.apply_pyfunc` that provides limited support for
  `tf.pyfunc`. Note that this is incompatible with serving. See documentation
  for more details.
* `CombinePerKey` now adds a dimension for the key.
* Depends on `numpy>=1.14.5,<2`.
* Depends on `apache-beam[gcp]>=2.10,<3`.
* Depends on `protobuf==3.7.0rc2`.
* `ExampleProtoCoder.encode` now converts a feature whose value is `None` to an
  empty value, where before it did not accept `None` as a valid value.
* `AnalyzeDataset`, `AnalyzeAndTransformDataset` and `TransformDataset` can now
  accept dictionaries which contain `None`, and which will be interpreted the
  same as an empty list.  They will never produce an output containing `None`.

## Breaking changes
* `ColumnSchema` and related classes (`Domain`, `Axis` and
  `ColumnRepresentation` and their subclasses) have been removed.  In order to
  create a schema, use `from_feature_spec`.  In order to inspect a schema
  use the `as_feature_spec` and `domains` methods of `Schema`.  The
  constructors of these classes are replaced by functions that still work when
  creating a `Schema` but this usage is deprecated.
* Requires pre-installed TensorFlow >=1.12,<2.
* `ExampleProtoCoder.decode` now converts a feature with empty value (e.g.
  `features { feature { key: "varlen" value { } } }`) or missing key for a
  feature (e.g. `features { }`) to a `None` in the output dictionary.  Before
  it would represent these with an empty list.  This better reflects the
  original example proto and is consistent with TensorFlow Data Validation.
* Coders now returns a `list` instead of an `ndarray` for a `VarLenFeature`.

## Deprecations

# Release 0.11.0

## Major Features and Improvements

## Bug Fixes and Other Changes
* 'tft.vocabulary' and 'tft.compute_and_apply_vocabulary' now support filtering
  based on mutual information when `labels` is provided.
* Export all package level exports of `tensorflow_transform`, from the
  `tensorflow_transform.beam` subpackage. This allows users to just import the
  `tensorflow_transform.beam` subpackage for all functionality.
* Adding API docs.
* Fix bug where Transform returned a different dtype for a VarLenFeature with
  0 elements.
* Depends on `apache-beam[gcp]>=2.8,<3`.

## Breaking changes
* Requires pre-installed TensorFlow >=1.11,<2.

## Deprecations
* All functions in `tensorflow_transform.saved.input_fn_maker` are deprecated.
  See the examples for how to construct the `input_fn` for training and serving.
  Note that the examples demonstrate the use of the `tf.estimator` API.  The
  functions named \*\_serving\_input\_fn were for use with the
  `tf.contrib.estimator` API which is now deprecated.  We do not provide
  examples of usage of the `tf.contrib.estimator` API, instead users should
  upgrade to the `tf.estimator` API.

# Release 0.9.0

## Major Features and Improvements
* Performance improvements for vocabulary generation when using top_k.
* Utility to deep-copy Beam `PCollection`s was added to avoid unnecessary
  materialization.
* Utilize deep_copy to avoid unnecessary materialization of pcollections when
  the input data is immutable. This feature is currently off by default and can
  be enabled by setting `tft.Context.use_deep_copy_optimization=True`.
* Add bucketize_per_key which computes separate quantiles for each key and then
  bucketizes each value according to the quantiles computed for its key.
* `tft.scale_to_z_score` is now implemented with a single pass over the data.
* Export schema_utils package to convert from the `tensorflow-metadata` package
  to the (soon to be deprecated) `tf_metadata` subpackage of
  `tensorflow-transform`.

## Bug Fixes and Other Changes
* Memory reduction during vocabulary generation.
* Clarify documentation on return values from `tft.compute_and_apply_vocabulary`
  and `tft.string_to_int`.
* `tft.unit` now explicitly creates Beam PCollections and validates the
  transformed dataset by writing and then reading it from disk.
* `tft.min`, `tft.size`, `tft.sum`, `tft.scale_to_z_score` and `tft.bucketize`
  now support `tf.SparseTensor`.
* Fix to `tft.scale_to_z_score` so it no longer attempts to divide by 0 when the
  variance is 0.
* Fix bug where internal graph analysis didn't handle the case where an
  operation has control inputs that are operations (as opposed to tensors).
* `tft.sparse_tensor_to_dense_with_shape` added which allows densifying a
  `SparseTensor` while specifying the resulting `Tensor`'s shape.
* Add `load_transform_graph` method to `TFTransformOutput` to load the transform
  graph without applying it.  This has the effect of adding variables to the
  checkpoint when calling it from the training `input_fn` when using
  `tf.Estimator`.
* 'tft.vocabulary' and 'tft.compute_and_apply_vocabulary' now accept an
  optional `weights` argument. When `weights` is provided, weighted frequencies
  are used instead of frequencies based on counts.
* 'tft.quantiles' and 'tft.bucketize' now accept an optional `weights` argument.
  When `weights` is provided, weighted count is used for quantiles instead of
  the counts themselves.
* Updated examples to construct the schema using
  `dataset_schema.from_feature_spec`.
* Updated the census example to allow the 'education-num' feature to be missing
  and fill in a default value when it is.
* Depends on `tensorflow-metadata>=0.9,<1`.
* Depends on `apache-beam[gcp]>=2.6,<3`.

## Breaking changes
* We now validate a `Schema` in its constructor to make sure that it can be
  converted to a feature spec.  In particular only `tf.int64`, `tf.string` and
  `tf.float32` types are allowed.
* We now disallow default values for `FixedColumnRepresentation`.
* It is no longer possible to set a default value in the Schema, and validation
  of shape parameters will occur earlier.
* Removed Schema.as_batched_placeholders() method.
* Removed all components of DatasetMetadata except the schema, and removed all
  related classes and code.
* Removed the merge method for DatasetMetadata and related classes.
* read_metadata can now only read from a single metadata directory and
  read_metadata and write_metadata no longer accept the `versions`  parameter.
  They now only read/write the JSON format.
* Requires pre-installed TensorFlow >=1.9,<2.

## Deprecations
* `apply_function` is no longer needed and is deprecated.
  `apply_function(fn, *args)` is now equivalent to `fn(*args)`.  tf.Transform
  is able to handle while loops and tables without the user wrapping the
  function call in `apply_function`.

# Release 0.8.0

## Major Features and Improvements
* Add TFTransformOutput utility class that wraps the output of tf.Transform for
  use in training.  This makes it easier to consume the output written by
  tf.Transform (see update examples for usage).
* Increase efficiency of `quantiles` (and therefore `bucketize`).

## Bug Fixes and Other Changes
* Change `tft.sum`/`tft.mean`/`tft.var` to only support basic numeric types.
* Widen the output type of `tft.sum` for some input types to avoid overflow
  and/or to preserve precision.
* For int32 and int64 input types, change the output type of `tft.mean`/
  `tft.var`/`tft.scale_to_z_score` from float64 to float32 .
* Change the output type of `tft.size` to be always int64.
* `Context` now accepts passthrough_keys which can be used when additional
  information should be attached to dataset instances in the pipeline which
  should not be part of the transformation graph, for example: instance keys.
* In addition to using TFTransformOutput, the examples demonstrate new workflows
  where a vocabulary is computed, but not applied, in the `preprocessing_fn`.
* Added dependency on the [absl-py package](https://pypi.org/project/absl-py/).
* `TransformTestCase` test cases can now be parameterized.
* Add support for partitioned variables when loading a model.
* Export the `coders` subpackage so that users can access it as `tft.coders`,
  e.g. `tft.coders.ExampleProtoCoder`.
* Setting dtypes for numpy arrays in `tft.coders.ExampleProtoCoder` and
  `tft.coders.CsvCoder`.
* `tft.mean`, `tft.max` and `tft.var` now support `tf.SparseTensor`.
* Update examples to use "core" TensorFlow estimator API (`tf.estimator`).
* Depends on `protobuf>=3.6.0<4`.

## Breaking changes
* `apply_saved_transform` is removed.  See note on
  `partially_apply_saved_transform` in the `Deprecations` section.
* No longer set `vocabulary_file` in `IntDomain` when using
  `tft.compute_and_apply_vocabulary` or `tft.apply_vocabulary`.
* Requires pre-installed TensorFlow >=1.8,<2.

## Deprecations
* The `expected_asset_file_contents` of
  `TransformTestCase.assertAnalyzeAndTransformResults` has been deprecated, use
  `expected_vocab_file_contents` instead.
* `transform_fn_io.TRANSFORMED_METADATA_DIR` and
  `transform_fn_io.TRANSFORM_FN_DIR` should not be used, they are now aliases
  for `TFTransformOutput.TRANSFORMED_METADATA_DIR` and
  `TFTransformOutput.TRANSFORM_FN_DIR` respectively.
* `partially_apply_saved_transform` is deprecated, users should use the
  `transform_raw_features` method of `TFTransformOutput` instead.  These differ
  in that `partially_apply_saved_transform` can also be used to return both the
  input placeholders and the outputs.  But users do not need this functionality
  because they will typically create the input placeholders themselves based
  on the feature spec.
* Renamed `tft.uniques` to `tft.vocabulary`, `tft.string_to_int` to
  `tft.compute_and_apply_vocabulary` and `tft.apply_vocab` to
  `tft.apply_vocabulary`.  The existing methods will remain for a few more minor
  releases but are now deprecated and should get migrated away from.

# Release 0.6.0

## Major Features and Improvements

## Bug Fixes and Other Changes
* Depends on `apache-beam[gcp]>=2.4,<3`.
* Trim min/max value in `tft.bucketize` where the computed number of bucket
  boundaries is more than requested. Updated documentation to clearly indicate
  that the number of buckets is computed using approximate algorithms, and that
  computed number can be more or less than requested.
* Change the namespace used for Beam metrics from `tensorflow_transform` to
  `tfx.Transform`.
* Update Beam metrics to also log vocabulary sizes.
* `CsvCoder` updated to support unicode.
* Update examples to not use the `coder` argument for IO, and instead use a
  separate `beam.Map` to encode/decode data.

## Breaking changes
* Requires pre-installed TensorFlow >=1.6,<2.

## Deprecations

# Release 0.5.0

## Major Features and Improvements
* Batching of input instances is now done automatically and dynamically.
* Added analyzers to compute covariance matrices (`tft.covariance`) and
  principal components for PCA (`tft.pca`).
* CombinerSpec and combine_analyzer now accept multiple inputs/outputs.

## Bug Fixes and Other Changes
* Depends on `apache-beam[gcp]>=2.3,<3`.
* Fixes a bug where TransformDataset would not return correct output if the
  output DatasetMetadata contained deferred values (such as vocabularies).
* Added checks that the prepreprocessing function's outputs all have the same
  size in the batch dimension.
* Added `tft.apply_buckets` which takes an input tensor and a list of bucket
  boundaries, and returns bucketized data.
* `tft.bucketize` and `tft.apply_buckets` now set metadata for the output
  tensor, which means the resulting tf.Metadata for the output of these
  functions will contain min and max values based on the number of buckets,
  and also be set to categorical.
* Testing helper function assertAnalyzeAndTransformResults can now also test
  the content of vocabulary files and other assets.
* Reduces the number of beam stages needed for certain analyzers, which can be
  a performance bottleneck when transforming many features.
* Performance improvements in `tft.uniques`.
* Fix a bug in `tft.bucketize` where the bucket boundary could be same as a
  min/max value, and was getting dropped.
* Allows scaling individual components of a tensor independently with
  `tft.scale_by_min_max`, `tft.scale_to_0_1`, and `tft.scale_to_z_score`.
* Fix a bug where `apply_saved_transform` could only be applied in the global
  name scope.
* Add warning when `frequency_threshold` that are <= 1.  This is a no-op and
  generally reflects mistaking `frequency_threshold` for a relative frequency
  where in fact it is an absolute frequency.

## Breaking changes
* The interfaces of CombinerSpec and combine_analyzer have changed to allow
  for multiple inputs/outputs.
* Requires pre-installed TensorFlow >=1.5,<2.

## Deprecations

# Release 0.4.0

## Major Features and Improvements
* Added a combine_analyzer() that supports user provided combiner, conforming to
  beam.CombinFn(). This allows users to implement custom combiners
  (e.g. median), to complement analyzers (like min, max) that are
  prepackaged in TFT.
* Quantiles Analyzer (`tft.quantiles`), with a corresponding `tft.bucketize`
  mapper.

## Bug Fixes and Other Changes
* Depends on `apache-beam[gcp]>=2.2,<3`.
* Fixes some KeyError issues that appeared in certain circumstances when one
  would call AnalyzeAndTransformDataset (due to a now-fixed Apache Beam [bug]
  (https://issues.apache.org/jira/projects/BEAM/issues/BEAM-2966)).
* Allow all functions that accept and return tensors, to accept an optional
  name scope, in line with TensorFlow coding conventions.
* Update examples to construct input functions by hand instead of using helper
  functions.
* Change scale_by_min_max/scale_to_0_1 to return the average(min, max) of the
  range in case all values are identical.
* Added export of serving model to examples.
* Use "core" version of feature columns (tf.feature_column instead of
  tf.contrib) in examples.
* A few bug fixes and improvements for coders regarding Python 3.

## Breaking changes
* Requires pre-installed TensorFlow >= 1.4.
* No longer distributing a WHL file in PyPI. Only doing a source distribution
  which should however be compatible with all platforms (ie you are still able
  to `pip install tensorflow-transform` and use `requirements.txt` or `setup.py`
  files for environment setup).
* Some functions now introduce a new name scope when they did not before so the
  names of tensors may change.  This will only affect you if you directly lookup
  tensors by name in the graph produced by tf.Transform.
* Various Analyzer Specs (\_NumericCombineSpec, \_UniquesSpec, \_QuantilesSpec)
  are now private. Analyzers are accessible only via the top-level TFT functions
  (min, max, sum, size, mean, var, uniques, quantiles).

## Deprecations
* The `serving_input_fn`s on `tensorflow_transform/saved/input_fn_maker.py` will
be removed on a future version and should not be used on new code,
see the `examples` directory for details on how to migrate your code to define
their own serving functions.

# Release 0.3.1

## Major Features and Improvements
* We now provide helper methods for creating `serving_input_receiver_fn` for use
with tf.estimator.  These mirror the existing functions targeting the
legacy tf.contrib.learn.estimators-- i.e. for each `*_serving_input_fn()`
in input_fn_maker there is now also a `*_serving_input_receiver_fn()`.

## Bug Fixes and Other Changes
* Introduced `tft.apply_vocab` this allows users to separately apply a single
  vocabulary (as generated by `tft.uniques`) to several different columns.
* Provide a source distribution tar `tensorflow-transform-X.Y.Z.tar.gz`.

## Breaking Changes
* The default prefix for `tft.string_to_int` `vocab_filename` changed from
`vocab_string_to_int` to `vocab_string_to_int_uniques`. To make your pipelines
resilient to implementation details please set `vocab_filename` if you are using
the generated vocab_filename on a downstream component.

# Release 0.3.0

## Major Features and Improvements
* Added hash_strings mapper.
* Write vocabularies as asset files instead of constants in the SavedModel.

## Bug Fixes and Other Changes
* 'tft.tfidf' now adds 1 to idf values so that terms in every document in the
  corpus have a non-zero tfidf value.
* Performance and memory usage improvement when running with Beam runners that
  use multi-threaded workers.
* Performance optimizations in ExampleProtoCoder.
* Depends on `apache-beam[gcp]>=2.1.1,<3`.
* Depends on `protobuf>=3.3<4`.
* Depends on `six>=1.9,<1.11`.

## Breaking Changes
* Requires pre-installed TensorFlow >= 1.3.
* Removed `tft.map` use `tft.apply_function` instead (as needed).
* Removed `tft.tfidf_weights` use `tft.tfidf` instead.
* `beam_metadata_io.WriteMetadata` now requires a second `pipeline` argument
  (see examples).
* A Beam bug will now affect users who call AnalyzeAndTransformDataset in
  certain circumstances.  Roughly speaking, if you call `beam.Pipeline()` at
  some point (as all our examples do) you will not experience this bug.  The
  bug is characterized by an error similar to
  `KeyError: (u'AnalyzeAndTransformDataset/AnalyzeDataset/ComputeTensorValues/Extract[Maximum:0]', None)`
  This [bug](https://issues.apache.org/jira/projects/BEAM/issues/BEAM-2966) will be fixed in Beam 2.2.

# Release 0.1.10

## Major Features and Improvements
* Add json-example serving input functions to TF.Transform.
* Add variance analyzer to tf.transform.

## Bug Fixes and Other Changes
* Remove duplication in output of `tft.tfidf`.
* Ensure ngrams output dense_shape is greater than or equal to 0.
* Alters the behavior and interface of tensorflow_transform.mappers.ngrams.
* Depends on `apache-beam[gcp]=>2,<3`.
* Making TF Parallelism runner-dependent.
* Fixes issue with csv serving input function.
* Various performance and stability improvements.

## Deprecations
* `tft.map` will be removed on version 0.2.0, see the `examples` directory for
  instructions on how to use `tft.apply_function` instead (as needed).
* `tft.tfidf_weights` will be removed on version 0.2.0, use `tft.tfidf` instead.

# Release 0.1.9

## Major Features and Improvements
* Refactor internals to remove Column and Statistic classes

## Bug Fixes and Other Changes
* Remove collections from graph to avoid warnings
* Return float32 from `tfidf_weights`
* Update tensorflow_transform to use `tf.saved_model` APIs.
* Add default values on example proto coder.
* Various performance and stability improvements.

