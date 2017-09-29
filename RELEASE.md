# Current version

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
* Depends on `protobuf>=3.3.0<4`.
* Depends on `six>=1.9,<1.11`.

## Breaking changes
* Requires pre-installed TensorFlow >= 1.3.
* Removed `tft.map`  use `tft.apply_function` instead (as needed).
* Removed `tft.tfidf_weights` use `tft.tfidf` instead.

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
