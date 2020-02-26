<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tft

Init module for TF.Transform.

## Modules

[`coders`](./tft/coders.md) module: Module level imports for tensorflow_transform.coders.

## Classes

[`class TFTransformOutput`](./tft/TFTransformOutput.md): A wrapper around the output of the tf.Transform.

[`class TransformFeaturesLayer`](./tft/TransformFeaturesLayer.md): A Keras layer for applying a tf.Transform output to input layers.

## Functions

[`apply_buckets(...)`](./tft/apply_buckets.md): Returns a bucketized column, with a bucket index assigned to each input.

[`apply_buckets_with_interpolation(...)`](./tft/apply_buckets_with_interpolation.md): Interpolates within the provided buckets and then normalizes to 0 to 1.

[`apply_function(...)`](./tft/apply_function.md): Deprecated function, equivalent to fn(*args). (deprecated)

[`apply_function_with_checkpoint(...)`](./tft/apply_function_with_checkpoint.md): Applies a tensor-in-tensor-out function with variables to some `Tensor`s.

[`apply_pyfunc(...)`](./tft/apply_pyfunc.md): Applies a python function to some `Tensor`s.

[`apply_saved_model(...)`](./tft/apply_saved_model.md): Applies a SavedModel to some `Tensor`s.

[`apply_vocab(...)`](./tft/apply_vocab.md): See <a href="./tft/apply_vocabulary.md"><code>tft.apply_vocabulary</code></a>. (deprecated)

[`apply_vocabulary(...)`](./tft/apply_vocabulary.md): Maps `x` to a vocabulary specified by the deferred tensor.

[`bag_of_words(...)`](./tft/bag_of_words.md): Computes a bag of "words" based on the specified ngram configuration.

[`bucketize(...)`](./tft/bucketize.md): Returns a bucketized column, with a bucket index assigned to each input.

[`bucketize_per_key(...)`](./tft/bucketize_per_key.md): Returns a bucketized column, with a bucket index assigned to each input.

[`compute_and_apply_vocabulary(...)`](./tft/compute_and_apply_vocabulary.md): Generates a vocabulary for `x` and maps it to an integer with this vocab.

[`count_per_key(...)`](./tft/count_per_key.md): Computes the count of each element of a `Tensor`.

[`covariance(...)`](./tft/covariance.md): Computes the covariance matrix over the whole dataset.

[`deduplicate_tensor_per_row(...)`](./tft/deduplicate_tensor_per_row.md): Deduplicates each row (0-th dimension) of the provided tensor.

[`estimated_probability_density(...)`](./tft/estimated_probability_density.md): Computes an approximate probability density at each x, given the bins.

[`get_analyze_input_columns(...)`](./tft/get_analyze_input_columns.md): Return columns that are required inputs of `AnalyzeDataset`.

[`get_transform_input_columns(...)`](./tft/get_transform_input_columns.md): Return columns that are required inputs of `TransformDataset`.

[`hash_strings(...)`](./tft/hash_strings.md): Hash strings into buckets.

[`histogram(...)`](./tft/histogram.md): Computes a histogram over x, given the bin boundaries or bin count.

[`max(...)`](./tft/max.md): Computes the maximum of the values of a `Tensor` over the whole dataset.

[`mean(...)`](./tft/mean.md): Computes the mean of the values of a `Tensor` over the whole dataset.

[`min(...)`](./tft/min.md): Computes the minimum of the values of a `Tensor` over the whole dataset.

[`ngrams(...)`](./tft/ngrams.md): Create a `SparseTensor` of n-grams.

[`pca(...)`](./tft/pca.md): Computes PCA on the dataset using biased covariance.

[`ptransform_analyzer(...)`](./tft/ptransform_analyzer.md): Applies a user-provided PTransform over the whole dataset.

[`quantiles(...)`](./tft/quantiles.md): Computes the quantile boundaries of a `Tensor` over the whole dataset.

[`scale_by_min_max(...)`](./tft/scale_by_min_max.md): Scale a numerical column into the range [output_min, output_max].

[`scale_by_min_max_per_key(...)`](./tft/scale_by_min_max_per_key.md): Scale a numerical column into a predefined range on a per-key basis.

[`scale_to_0_1(...)`](./tft/scale_to_0_1.md): Returns a column which is the input column scaled to have range [0,1].

[`scale_to_0_1_per_key(...)`](./tft/scale_to_0_1_per_key.md): Returns a column which is the input column scaled to have range [0,1].

[`scale_to_z_score(...)`](./tft/scale_to_z_score.md): Returns a standardized column with mean 0 and variance 1.

[`scale_to_z_score_per_key(...)`](./tft/scale_to_z_score_per_key.md): Returns a standardized column with mean 0 and variance 1, grouped per key.

[`segment_indices(...)`](./tft/segment_indices.md): Returns a `Tensor` of indices within each segment.

[`size(...)`](./tft/size.md): Computes the total size of instances in a `Tensor` over the whole dataset.

[`sparse_tensor_to_dense_with_shape(...)`](./tft/sparse_tensor_to_dense_with_shape.md): Converts a `SparseTensor` into a dense tensor and sets its shape.

[`string_to_int(...)`](./tft/string_to_int.md): See <a href="./tft/compute_and_apply_vocabulary.md"><code>tft.compute_and_apply_vocabulary</code></a>. (deprecated)

[`sum(...)`](./tft/sum.md): Computes the sum of the values of a `Tensor` over the whole dataset.

[`tfidf(...)`](./tft/tfidf.md): Maps the terms in x to their term frequency * inverse document frequency.

[`uniques(...)`](./tft/uniques.md): See <a href="./tft/vocabulary.md"><code>tft.vocabulary</code></a>. (deprecated)

[`var(...)`](./tft/var.md): Computes the variance of the values of a `Tensor` over the whole dataset.

[`vocabulary(...)`](./tft/vocabulary.md): Computes the unique values of a `Tensor` over the whole dataset.

[`word_count(...)`](./tft/word_count.md): Find the token count of each document/row.

