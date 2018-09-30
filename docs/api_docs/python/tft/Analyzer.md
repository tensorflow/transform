<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.Analyzer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="attributes"/>
<meta itemprop="property" content="control_inputs"/>
<meta itemprop="property" content="inputs"/>
<meta itemprop="property" content="outputs"/>
<meta itemprop="property" content="__init__"/>
</div>

# tft.Analyzer

## Class `Analyzer`



A class representing computation that will be done by Beam.

An Analyzer is like a tf.Operation except that it requires computation over
the full dataset.  E.g. sum(my_tensor) will compute the sum of the value of
my_tensor over all instances in the dataset.  The Analyzer class contains the
inputs to this computation, and placeholders which will later be converted to
constants during a call to AnalyzeDataset.

Analyzer implementations write some files to disk in a temporary location and
return tensors that contain the filename.  These outputs must be added to the
tf.GraphKeys.ASSET_FILEPATHS collection.  Doing so will ensure a few things
happen:
* the tensor will be removed from the collection prior to writing the
  SavedModel (since the tensor will be replaced)
* when the tensor is replaced, the replacement will be added to the
  tf.GraphKeys.ASSET_FILEPATHS collection
* This in turn causes the underlying file to be added to the SavedModel's
  `assets` directory when the model is saved

#### Args:

* <b>`inputs`</b>: The `Tensor`s that are used to create inputs to this analyzer,
* <b>`outputs`</b>: The `Tensor`s whose values will be replaced by the result of the
      analyzer.
* <b>`attributes`</b>: An object that will be used to determine how the analyzer is
      implemented by Beam.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    inputs,
    outputs,
    attributes
)
```





## Properties

<h3 id="attributes"><code>attributes</code></h3>



<h3 id="control_inputs"><code>control_inputs</code></h3>



<h3 id="inputs"><code>inputs</code></h3>



<h3 id="outputs"><code>outputs</code></h3>





