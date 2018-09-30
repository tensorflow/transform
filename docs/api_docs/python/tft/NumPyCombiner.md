<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.NumPyCombiner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_input"/>
<meta itemprop="property" content="create_accumulator"/>
<meta itemprop="property" content="extract_output"/>
<meta itemprop="property" content="merge_accumulators"/>
<meta itemprop="property" content="num_outputs"/>
</div>

# tft.NumPyCombiner

## Class `NumPyCombiner`



Combines the PCollection only on the 0th dimension using nparray.

#### Args:

* <b>`fn`</b>: The numpy function representing the reduction to be done.
* <b>`output_dtypes`</b>: The numpy dtype to cast each output to.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    fn,
    output_dtypes
)
```





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



<h3 id="num_outputs"><code>num_outputs</code></h3>

``` python
num_outputs()
```

Return the number of outputs that are produced by extract_output.

Returns: The number of outputs extract_output will produce.



