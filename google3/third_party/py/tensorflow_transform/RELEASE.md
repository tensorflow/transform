# Release 0.1.10

## Major Features and Improvements
* Add json-example serving input functions to TF.Transform.
* Add variance analyzer to tf.transform.

## Bug Fixes and Other Changes
* Remove duplication in output of `tft.tfidf`.
* Ensure ngrams output dense_shape is greater than or equal to 0.
* Alters the behavior and interface of tensorflow_transform.mappers.ngrams.
* Use `apache-beam[gcp] >=2,<3`
* Making TF Parallelism runner-dependent.
* Fixes issue with csv serving input function.

## Deprecations
* `tft.map` will be removed on version 0.2.0, see the `examples` directory for
  instructions on how to use `tft.apply_function` instead (as needed).
* `tft.tfidf_weights` will be removed on version 0.2.0, use `tft.tfidf` instead.

# Release 0.1.9

## Major Features and Improvements
* Refactor internals to remove Column and Statistic classes

## Bug Fixes and Other Changes
* Remove collections from graph to avoid warnings
* Return float32 from tfidf_weights
* Update tensorflow_transform to use tf.saved_model APIs.
* Add default values on example proto coder.
