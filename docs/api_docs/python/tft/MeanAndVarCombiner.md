<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.MeanAndVarCombiner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="accumulator_coder"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_input"/>
<meta itemprop="property" content="create_accumulator"/>
<meta itemprop="property" content="extract_output"/>
<meta itemprop="property" content="merge_accumulators"/>
<meta itemprop="property" content="output_tensor_infos"/>
</div>

# tft.MeanAndVarCombiner

## Class `MeanAndVarCombiner`



Combines a PCollection of accumulators to compute mean and variance.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    output_numpy_dtype,
    output_shape=None
)
```





## Properties

<h3 id="accumulator_coder"><code>accumulator_coder</code></h3>





## Methods

<h3 id="add_input"><code>add_input</code></h3>

``` python
add_input(
    accumulator,
    batch_values
)
```

Composes an accumulator from batch_values and calls merge_accumulators.

#### Args:

* <b>`accumulator`</b>: The `_MeanAndVarAccumulator` computed so far.
* <b>`batch_values`</b>: A `_MeanAndVarAccumulator` for the current batch.


#### Returns:

A `_MeanAndVarAccumulator` which is accumulator and batch_values combined.

<h3 id="create_accumulator"><code>create_accumulator</code></h3>

``` python
create_accumulator()
```

Create an accumulator with all zero entries.

<h3 id="extract_output"><code>extract_output</code></h3>

``` python
extract_output(accumulator)
```

Converts an accumulator into the output (mean, var) tuple.

#### Args:

* <b>`accumulator`</b>: the final `_MeanAndVarAccumulator` value.


#### Returns:

A 2-tuple composed of (mean, var) or None if accumulator is None.

<h3 id="merge_accumulators"><code>merge_accumulators</code></h3>

``` python
merge_accumulators(accumulators)
```

Merges several `_MeanAndVarAccumulator`s to a single accumulator.

#### Args:

* <b>`accumulators`</b>: A list of `_MeanAndVarAccumulator`s and/or Nones.


#### Returns:

The sole merged `_MeanAndVarAccumulator`.

<h3 id="output_tensor_infos"><code>output_tensor_infos</code></h3>

``` python
output_tensor_infos()
```





