<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="VOCAB_FILENAME_PREFIX"/>
<meta itemprop="property" content="VOCAB_FREQUENCY_FILENAME_PREFIX"/>
</div>

# Module: tft

Init module for TF.Transform.

## Modules

[`coders`](./tft/coders.md) module: Module level imports for tensorflow_transform.coders.

## Classes

[`class CovarianceCombiner`](./tft/CovarianceCombiner.md): Combines the PCollection to compute the biased covariance matrix.

[`class MeanAndVarCombiner`](./tft/MeanAndVarCombiner.md): Combines a PCollection of accumulators to compute mean and variance.

[`class NumPyCombiner`](./tft/NumPyCombiner.md): Combines the PCollection only on the 0th dimension using nparray.

[`class PCACombiner`](./tft/PCACombiner.md): Compute PCA of accumulated data using the biased covariance matrix.

[`class QuantilesCombiner`](./tft/QuantilesCombiner.md): Computes quantiles on the PCollection.

[`class TFTransformOutput`](./tft/TFTransformOutput.md): A wrapper around the output of the tf.Transform.

## Functions

[`apply_analyzer(...)`](./tft/apply_analyzer.md): Applies the analyzer over the whole dataset.

[`apply_buckets(...)`](./tft/apply_buckets.md): Returns a bucketized column, with a bucket index assigned to each input.

[`apply_function(...)`](./tft/apply_function.md): Deprecated function, equivalent to fn(*args). (deprecated)

[`apply_function_with_checkpoint(...)`](./tft/apply_function_with_checkpoint.md): Applies a tensor-in-tensor-out function with variables to some `Tensor`s.

[`apply_saved_model(...)`](./tft/apply_saved_model.md): Applies a SavedModel to some `Tensor`s.

[`apply_vocab(...)`](./tft/apply_vocab.md): See <a href="./tft/apply_vocabulary.md"><code>tft.apply_vocabulary</code></a>. (deprecated)

[`apply_vocabulary(...)`](./tft/apply_vocabulary.md): Maps `x` to a vocabulary specified by the deferred tensor.

[`bucketize(...)`](./tft/bucketize.md): Returns a bucketized column, with a bucket index assigned to each input.

[`bucketize_per_key(...)`](./tft/bucketize_per_key.md): Returns a bucketized column, with a bucket index assigned to each input.

[`compute_and_apply_vocabulary(...)`](./tft/compute_and_apply_vocabulary.md): Generates a vocabulary for `x` and maps it to an integer with this vocab.

[`covariance(...)`](./tft/covariance.md): Computes the covariance matrix over the whole dataset.

[`hash_strings(...)`](./tft/hash_strings.md): Hash strings into buckets.

[`max(...)`](./tft/max.md): Computes the maximum of the values of a `Tensor` over the whole dataset.

[`mean(...)`](./tft/mean.md): Computes the mean of the values of a `Tensor` over the whole dataset.

[`min(...)`](./tft/min.md): Computes the minimum of the values of a `Tensor` over the whole dataset.

[`ngrams(...)`](./tft/ngrams.md): Create a `SparseTensor` of n-grams.

[`pca(...)`](./tft/pca.md): Computes pca on the dataset using biased covariance.

[`ptransform_analyzer(...)`](./tft/ptransform_analyzer.md): Applies a user-provided PTransform over the whole dataset.

[`quantiles(...)`](./tft/quantiles.md): Computes the quantile boundaries of a `Tensor` over the whole dataset.

[`sanitized_vocab_filename(...)`](./tft/sanitized_vocab_filename.md): Generates a sanitized filename either from the given filename or the scope.

[`scale_by_min_max(...)`](./tft/scale_by_min_max.md): Scale a numerical column into the range [output_min, output_max].

[`scale_to_0_1(...)`](./tft/scale_to_0_1.md): Returns a column which is the input column scaled to have range [0,1].

[`scale_to_z_score(...)`](./tft/scale_to_z_score.md): Returns a standardized column with mean 0 and variance 1.

[`segment_indices(...)`](./tft/segment_indices.md): Returns a `Tensor` of indices within each segment.

[`size(...)`](./tft/size.md): Computes the total size of instances in a `Tensor` over the whole dataset.

[`sparse_tensor_to_dense_with_shape(...)`](./tft/sparse_tensor_to_dense_with_shape.md): Converts a `SparseTensor` into a dense tensor and sets its shape.

[`string_to_int(...)`](./tft/string_to_int.md): See <a href="./tft/compute_and_apply_vocabulary.md"><code>tft.compute_and_apply_vocabulary</code></a>. (deprecated)

[`sum(...)`](./tft/sum.md): Computes the sum of the values of a `Tensor` over the whole dataset.

[`tfidf(...)`](./tft/tfidf.md): Maps the terms in x to their term frequency * inverse document frequency.

[`uniques(...)`](./tft/uniques.md): See <a href="./tft/vocabulary.md"><code>tft.vocabulary</code></a>. (deprecated)

[`var(...)`](./tft/var.md): Computes the variance of the values of a `Tensor` over the whole dataset.

[`vocabulary(...)`](./tft/vocabulary.md): Computes the unique values of a `Tensor` over the whole dataset.

## Other Members

<h3 id="VOCAB_FILENAME_PREFIX"><code>VOCAB_FILENAME_PREFIX</code></h3>

<h3 id="VOCAB_FREQUENCY_FILENAME_PREFIX"><code>VOCAB_FREQUENCY_FILENAME_PREFIX</code></h3>

