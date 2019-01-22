<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.NumPyCombiner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="accumulator_coder"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_input"/>
<meta itemprop="property" content="create_accumulator"/>
<meta itemprop="property" content="extract_output"/>
<meta itemprop="property" content="merge_accumulators"/>
<meta itemprop="property" content="output_tensor_infos"/>
</div>

# tft.NumPyCombiner

## Class `NumPyCombiner`



Combines the PCollection only on the 0th dimension using nparray.

#### Args:

* <b>`fn`</b>: The numpy function representing the reduction to be done.
* <b>`output_dtypes`</b>: The numpy dtype to cast each output to.
* <b>`output_shapes`</b>: The shapes of the outputs.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    fn,
    output_dtypes,
    output_shapes
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



<h3 id="create_accumulator"><code>create_accumulator</code></h3>

``` python
create_accumulator()
```



<h3 id="extract_output"><code>extract_output</code></h3>

``` python
extract_output(accumulator)
```



<h3 id="merge_accumulators"><code>merge_accumulators</code></h3>

``` python
merge_accumulators(accumulators)
```



<h3 id="output_tensor_infos"><code>output_tensor_infos</code></h3>

``` python
output_tensor_infos()
```





