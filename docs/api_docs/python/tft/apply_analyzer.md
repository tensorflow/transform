<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_analyzer" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_analyzer

``` python
tft.apply_analyzer(
    analyzer_def_cls,
    *tensor_inputs,
    **analyzer_def_kwargs
)
```

Applies the analyzer over the whole dataset.

#### Args:

* <b>`analyzer_def_cls`</b>: A class inheriting from analyzer_nodes.AnalyzerDef that
    should be applied.
* <b>`*tensor_inputs`</b>: A list of input `Tensor`s or `SparseTensor`s.
* <b>`**analyzer_def_kwargs`</b>: KW arguments to use when constructing
    analyzer_def_cls.


#### Returns:

A list of `Tensor`s representing the values of the analysis result.