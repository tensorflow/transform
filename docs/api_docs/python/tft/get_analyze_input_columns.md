<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.get_analyze_input_columns" />
<meta itemprop="path" content="Stable" />
</div>

# tft.get_analyze_input_columns

``` python
tft.get_analyze_input_columns(
    preprocessing_fn,
    feature_spec
)
```

Return columns that are required inputs of `AnalyzeDataset`.

#### Args:

* <b>`preprocessing_fn`</b>: A tf.transform preprocessing_fn.
* <b>`feature_spec`</b>: A dict of feature name to feature specification.


#### Returns:

A list of columns that are required inputs of analyzers.