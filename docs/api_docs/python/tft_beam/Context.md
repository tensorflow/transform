<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft_beam.Context" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_base_temp_dir"/>
<meta itemprop="property" content="get_desired_batch_size"/>
<meta itemprop="property" content="get_passthrough_keys"/>
<meta itemprop="property" content="get_use_deep_copy_optimization"/>
</div>

# tft_beam.Context

## Class `Context`



Context manager for tensorflow-transform.

All the attributes in this context are kept on a thread local state.

#### Args:

* <b>`temp_dir`</b>: (Optional) The temporary directory used within in this block.
* <b>`desired_batch_size`</b>: (Optional) A batch size to batch elements by. If not
      provided, a batch size will be computed automatically.
* <b>`passthrough_keys`</b>: (Optional) A set of strings that are keys to
      instances that should pass through the pipeline and be hidden from
      the preprocessing_fn. This should only be used in cases where additional
      information should be attached to instances in the pipeline which should
      not be part of the transformation graph, instance keys is one such
      example.

Note that the temp dir should be accessible to worker jobs, e.g. if running
with the Cloud Dataflow runner, the temp dir should be on GCS and should have
permissions that allow both launcher and workers to access it.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    temp_dir=None,
    desired_batch_size=None,
    passthrough_keys=None,
    use_deep_copy_optimization=None
)
```





## Methods

<h3 id="__enter__"><code>__enter__</code></h3>

``` python
__enter__()
```



<h3 id="__exit__"><code>__exit__</code></h3>

``` python
__exit__(*exn_info)
```



<h3 id="create_base_temp_dir"><code>create_base_temp_dir</code></h3>

``` python
@classmethod
create_base_temp_dir(cls)
```

Generate a temporary location.

<h3 id="get_desired_batch_size"><code>get_desired_batch_size</code></h3>

``` python
@classmethod
get_desired_batch_size(cls)
```

Retrieves a user set fixed batch size, None if not set.

<h3 id="get_passthrough_keys"><code>get_passthrough_keys</code></h3>

``` python
@classmethod
get_passthrough_keys(cls)
```

Retrieves a user set passthrough_keys, None if not set.

<h3 id="get_use_deep_copy_optimization"><code>get_use_deep_copy_optimization</code></h3>

``` python
@classmethod
get_use_deep_copy_optimization(cls)
```

Retrieves a user set use_deep_copy_optimization, None if not set.



