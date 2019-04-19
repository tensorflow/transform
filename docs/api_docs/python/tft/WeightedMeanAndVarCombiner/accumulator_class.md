<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.WeightedMeanAndVarCombiner.accumulator_class" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="count"/>
<meta itemprop="property" content="mean"/>
<meta itemprop="property" content="variance"/>
<meta itemprop="property" content="weight"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="make_nan_to_num"/>
</div>

# tft.WeightedMeanAndVarCombiner.accumulator_class

## Class `accumulator_class`



Container for WeightedMeanAndVarCombiner intermediate values.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    count,
    mean,
    variance,
    weight
)
```

Create new instance of WeightedMeanAndVarAccumulator(count, mean, variance, weight)



## Properties

<h3 id="count"><code>count</code></h3>



<h3 id="mean"><code>mean</code></h3>



<h3 id="variance"><code>variance</code></h3>



<h3 id="weight"><code>weight</code></h3>





## Methods

<h3 id="make_nan_to_num"><code>make_nan_to_num</code></h3>

``` python
@classmethod
make_nan_to_num(
    cls,
    counts,
    means,
    variances,
    weights
)
```





