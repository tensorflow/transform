<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft_beam.info_theory.calculate_partial_mutual_information" />
<meta itemprop="path" content="Stable" />
</div>

# tft_beam.info_theory.calculate_partial_mutual_information

``` python
tft_beam.info_theory.calculate_partial_mutual_information(
    n_ij,
    x_i,
    y_j,
    n
)
```

Calculates Mutual Information for x=i, y=j from sample counts.

The standard formulation of mutual information is:
MI(X,Y) = Sum_i,j {p_ij * log2(p_ij / p_i * p_j)}
We are operating over counts (p_ij = n_ij / n), so this is transformed into
MI(X,Y) = Sum_i,j {n_ij * (log2(n_ij) + log2(n) - log2(x_i) - log2(y_j))} / n
This function returns the argument to the summation, the mutual information
for a particular pair of values x_i, y_j (the caller is expected to divide
the summation by n to compute the final mutual information result).

#### Args:

* <b>`n_ij`</b>: The co-occurrence of x=i and y=j
* <b>`x_i`</b>: The frequency of x=i.
* <b>`y_j`</b>: The frequency of y=j.
* <b>`n`</b>: The total # observations


#### Returns:

Mutual information for the cell x=i, y=j.