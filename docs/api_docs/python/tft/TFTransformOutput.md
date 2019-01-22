<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.TFTransformOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="post_transform_statistics_path"/>
<meta itemprop="property" content="pre_transform_statistics_path"/>
<meta itemprop="property" content="raw_metadata"/>
<meta itemprop="property" content="transform_savedmodel_dir"/>
<meta itemprop="property" content="transformed_metadata"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="load_transform_graph"/>
<meta itemprop="property" content="num_buckets_for_transformed_feature"/>
<meta itemprop="property" content="raw_feature_spec"/>
<meta itemprop="property" content="transform_raw_features"/>
<meta itemprop="property" content="transformed_feature_spec"/>
<meta itemprop="property" content="vocabulary_by_name"/>
<meta itemprop="property" content="vocabulary_file_by_name"/>
<meta itemprop="property" content="vocabulary_size_by_name"/>
<meta itemprop="property" content="POST_TRANSFORM_FEATURE_STATS_PATH"/>
<meta itemprop="property" content="PRE_TRANSFORM_FEATURE_STATS_PATH"/>
<meta itemprop="property" content="RAW_METADATA_DIR"/>
<meta itemprop="property" content="TRANSFORMED_METADATA_DIR"/>
<meta itemprop="property" content="TRANSFORM_FN_DIR"/>
</div>

# tft.TFTransformOutput

## Class `TFTransformOutput`



A wrapper around the output of the tf.Transform.

#### Args:

* <b>`transform_output_dir`</b>: The directory containig tf.Transform output.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(transform_output_dir)
```





## Properties

<h3 id="post_transform_statistics_path"><code>post_transform_statistics_path</code></h3>

Returns the path to the post-transform datum statistics.

Note: post_transform_statistics is not guaranteed to exist in the output of
tf.transform and hence using this could fail, if post_transform statistics
is not present in TFTransformOutput.

<h3 id="pre_transform_statistics_path"><code>pre_transform_statistics_path</code></h3>

Returns the path to the pre-transform datum statistics.

Note: pre_transform_statistics is not guaranteed to exist in the output of
tf.transform and hence using this could fail, if pre_transform statistics is
not present in TFTransformOutput.

<h3 id="raw_metadata"><code>raw_metadata</code></h3>

A DatasetMetadata.

Note: raw_metadata is not guaranteed to exist in the output of tf.transform
and hence using this could fail, if raw_metadata is not present in
TFTransformOutput.

<h3 id="transform_savedmodel_dir"><code>transform_savedmodel_dir</code></h3>

A python str.

<h3 id="transformed_metadata"><code>transformed_metadata</code></h3>

A DatasetMetadata.



## Methods

<h3 id="load_transform_graph"><code>load_transform_graph</code></h3>

``` python
load_transform_graph()
```

Load the transform graph without replacing any placeholders.

This is necessary to ensure that variables in the transform graph are
included in the training checkpoint when using tf.Estimator.  This should
be called in the training input_fn.

<h3 id="num_buckets_for_transformed_feature"><code>num_buckets_for_transformed_feature</code></h3>

``` python
num_buckets_for_transformed_feature(name)
```

Returns the number of buckets for an integerized transformed feature.

<h3 id="raw_feature_spec"><code>raw_feature_spec</code></h3>

``` python
raw_feature_spec()
```

Returns a feature_spec for the raw features.

#### Returns:

A dict from feature names to FixedLenFeature/SparseFeature/VarLenFeature.

<h3 id="transform_raw_features"><code>transform_raw_features</code></h3>

``` python
transform_raw_features(raw_features)
```

Takes a dict of tensors representing raw features and transforms them.

Takes a dictionary of `Tensor`s or `SparseTensor`s that represent the raw
features, and applies the transformation defined by tf.Transform.

#### Args:

* <b>`raw_features`</b>: A dict whose keys are feature names and values are `Tensor`s
      or `SparseTensor`s.


#### Returns:

A dict whose keys are feature names and values are `Tensor`s or
    `SparseTensor`s representing transformed features.

<h3 id="transformed_feature_spec"><code>transformed_feature_spec</code></h3>

``` python
transformed_feature_spec()
```

Returns a feature_spec for the transformed features.

#### Returns:

A dict from feature names to FixedLenFeature/SparseFeature/VarLenFeature.

<h3 id="vocabulary_by_name"><code>vocabulary_by_name</code></h3>

``` python
vocabulary_by_name(vocab_filename)
```

Like vocabulary_file_by_name but returns a list.

<h3 id="vocabulary_file_by_name"><code>vocabulary_file_by_name</code></h3>

``` python
vocabulary_file_by_name(vocab_filename)
```

Returns the vocabulary file path created in the preprocessing function.

`vocab_filename` must be the name used as the vocab_filename argument to
tft.compute_and_apply_vocabulary or tft.vocabulary. By convention, this
should be the name of the feature that the vocab was computed for, where
possible.

#### Args:

* <b>`vocab_filename`</b>: The relative filename to lookup.

<h3 id="vocabulary_size_by_name"><code>vocabulary_size_by_name</code></h3>

``` python
vocabulary_size_by_name(vocab_filename)
```

Like vocabulary_file_by_name, but returns the size of vocabulary.



## Class Members

<h3 id="POST_TRANSFORM_FEATURE_STATS_PATH"><code>POST_TRANSFORM_FEATURE_STATS_PATH</code></h3>

<h3 id="PRE_TRANSFORM_FEATURE_STATS_PATH"><code>PRE_TRANSFORM_FEATURE_STATS_PATH</code></h3>

<h3 id="RAW_METADATA_DIR"><code>RAW_METADATA_DIR</code></h3>

<h3 id="TRANSFORMED_METADATA_DIR"><code>TRANSFORMED_METADATA_DIR</code></h3>

<h3 id="TRANSFORM_FN_DIR"><code>TRANSFORM_FN_DIR</code></h3>

