<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.PCACombiner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="accumulator_coder"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_input"/>
<meta itemprop="property" content="create_accumulator"/>
<meta itemprop="property" content="extract_output"/>
<meta itemprop="property" content="merge_accumulators"/>
<meta itemprop="property" content="output_tensor_infos"/>
</div>

# tft.PCACombiner

## Class `PCACombiner`

Inherits From: [`CovarianceCombiner`](../tft/CovarianceCombiner.md)

Compute PCA of accumulated data using the biased covariance matrix.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    output_dim=None,
    numpy_dtype=np.float64,
    output_shape=None
)
```

Store pca output dimension, and dtype for precision.



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

Compute sum of input cross-terms, sum of inputs, and count.

The cross terms for a numeric 1d array x are given by the set:
{z_ij = x_i * x_j for all indices i and j}. This is stored as a 2d array.
Since next_input is an array of 1d numeric arrays (i.e. a 2d array),
matmul(transpose(next_input), next_input) will automatically sum up
the cross terms of each 1d array in next_input.

#### Args:

* <b>`accumulator`</b>: running sum of cross terms, input vectors, and count
* <b>`batch_values`</b>: entries from the pipeline, which must be single element list
      containing a 2d array
  representing multiple 1d arrays


#### Returns:

An accumulator with next_input considered in its running list of
sum_product, sum_vectors, and count of input rows.

<h3 id="create_accumulator"><code>create_accumulator</code></h3>

``` python
create_accumulator()
```

Create an accumulator with all zero entries.

<h3 id="extract_output"><code>extract_output</code></h3>

``` python
extract_output(accumulator)
```

Compute PCA of the accumulated data using the biased covariance matrix.

Following the covariance computation in CovarianceCombiner, this method runs
eigenvalue decomposition on the covariance matrix, sorts eigenvalues in
decreasing order, and returns the first output_dim corresponding
eigenvectors (principal components) as a matrix.

#### Args:

* <b>`accumulator`</b>: final accumulator as a list of the sum of cross-terms matrix,
    sum of input vectors, and count.


#### Returns:

A list containing a matrix of shape (input_dim, output_dim).

<h3 id="merge_accumulators"><code>merge_accumulators</code></h3>

``` python
merge_accumulators(accumulators)
```

Sums values in each accumulator entry.

<h3 id="output_tensor_infos"><code>output_tensor_infos</code></h3>

``` python
output_tensor_infos()
```





