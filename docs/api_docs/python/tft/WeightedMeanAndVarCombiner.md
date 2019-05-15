<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.WeightedMeanAndVarCombiner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="accumulator_class"/>
<meta itemprop="property" content="accumulator_coder"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_input"/>
<meta itemprop="property" content="compute_running_update"/>
<meta itemprop="property" content="create_accumulator"/>
<meta itemprop="property" content="extract_output"/>
<meta itemprop="property" content="merge_accumulators"/>
<meta itemprop="property" content="output_tensor_infos"/>
</div>

# tft.WeightedMeanAndVarCombiner

## Class `WeightedMeanAndVarCombiner`



Combines a PCollection of accumulators to compute mean and variance.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    output_numpy_dtype,
    output_shape=None,
    compute_variance=True,
    compute_weighted=False
)
```

Init method for WeightedMeanAndVarCombiner.

#### Args:

* <b>`output_numpy_dtype`</b>: A numpy dtype that the outputs are cast to.
* <b>`output_shape`</b>: The shape of the resulting Tensors.
* <b>`compute_variance`</b>: A bool indicating whether or not a variance should be
    calculated and returned.
* <b>`compute_weighted`</b>: A bool indicating whether or not weights are provided
    and all calculations should be weighted.



## Child Classes
[`class accumulator_class`](../tft/WeightedMeanAndVarCombiner/accumulator_class.md)

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

* <b>`accumulator`</b>: The `_WeightedMeanAndVarAccumulator` computed so far.
* <b>`batch_values`</b>: A `_WeightedMeanAndVarAccumulator` for the current batch.


#### Returns:

A `_WeightedMeanAndVarAccumulator` which is accumulator and batch_values
  combined.

<h3 id="compute_running_update"><code>compute_running_update</code></h3>

``` python
compute_running_update(
    total_count,
    current_count,
    update
)
```

Numerically stable way of computing a streaming batched update.

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

* <b>`accumulator`</b>: the final `_WeightedMeanAndVarAccumulator` value.


#### Returns:

A 2-tuple composed of (mean, var).

<h3 id="merge_accumulators"><code>merge_accumulators</code></h3>

``` python
merge_accumulators(accumulators)
```

Merges several `_WeightedMeanAndVarAccumulator`s to a single accumulator.

#### Args:

* <b>`accumulators`</b>: A list of `_WeightedMeanAndVarAccumulator`s.


#### Returns:

The sole merged `_WeightedMeanAndVarAccumulator`.

<h3 id="output_tensor_infos"><code>output_tensor_infos</code></h3>

``` python
output_tensor_infos()
```





