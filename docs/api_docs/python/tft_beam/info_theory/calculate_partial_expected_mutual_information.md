<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft_beam.info_theory.calculate_partial_expected_mutual_information" />
<meta itemprop="path" content="Stable" />
</div>

# tft_beam.info_theory.calculate_partial_expected_mutual_information

``` python
tft_beam.info_theory.calculate_partial_expected_mutual_information(
    n,
    x_i,
    y_j
)
```

Calculates the partial expected mutual information (EMI) of two variables.

  EMI reflects the MI expected by chance, and is used to compute adjusted
  mutual information. See www.wikipedia.org/wiki/Adjusted_mutual_information.

  The EMI for two variables x and y, is the sum of the expected mutual info
  for each value of x with each value of y. This function computes the EMI
  for a single value of each variable (x_i, y_j) and is thus considered a
  partial EMI calculation.

  Specifically:
  EMI(x, y) = sum_{n_ij = max(0, x_i + y_j - n) to min(x_i, y_j)} (
    n_ij / n * log2((n * n_ij / (x_i * y_j))
    * ((x_i! * y_j! * (n - x_i)! * (n - y_j)!) /
    (n! * n_ij! * (x_i - n_ij)! * (y_j - n_ij)! * (n - x_i - y_j + n_ij)!)))
  where n_ij is the joint count of x taking on value i and y taking on
  value j, x_i is the count for x taking on value i, y_j is the count for y
  taking on value j, and n represents total count.

#### Args:

* <b>`n`</b>: The sum of weights for all values.
* <b>`x_i`</b>: The sum of weights for the first variable taking on value i
* <b>`y_j`</b>: The sum of weights for the second variable taking on value j


#### Returns:

Calculated expected mutual information for x_i, y_j.