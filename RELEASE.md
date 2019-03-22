<!-- mdformat off(mdformat causes unwanted indentation changes) -->
# Current version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes
* `sparse_tensor_to_dense_with_shape` now accepts an optional `default_value`
  parameter.
* `tft.vocabulary` and `tft.compute_and_apply_vocabulary` now support
  `fingerprint_shuffle` to sort the vocabularies by fingerprint instead of
  counts. This is useful for load balancing the training parameter servers.
  This is an experimental feature.
* Fix numerical instability in `tft.vocabulary` mutual information calculations
* `tft.vocabulary` and `tft.compute_and_apply_vocabulary` now support computing
  vocabularies over integer categoricals
* New numeric normalization method available:
  `tft.apply_buckets_with_interpolation`

## Breaking changes

## Deprecations

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
* 'tft.quantiles' and 'tft.bucketize' now accept an optoinal `weights` argument.
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
  `transform_raw_features` method of `TFTransformOuptut` instead.  These differ
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
