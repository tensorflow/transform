<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.coders.ExampleProtoCoder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="decode"/>
<meta itemprop="property" content="encode"/>
</div>

# tft.coders.ExampleProtoCoder

## Class `ExampleProtoCoder`



A coder between maybe-serialized TF Examples and tf.Transform datasets.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    schema,
    serialized=True
)
```

Build an ExampleProtoCoder.

#### Args:

* <b>`schema`</b>: A `Schema` object.
* <b>`serialized`</b>: Whether to encode / decode serialized Example protos (as
    opposed to in-memory Example protos). The default (True) is used for
    backwards compatibility. Note that the serialized=True option might be
    removed in a future version.

#### Raises:

* <b>`ValueError`</b>: If `schema` is invalid.



## Methods

<h3 id="decode"><code>decode</code></h3>

``` python
decode(example_proto)
```

Decode tf.Example as a tf.transform encoded dict.

<h3 id="encode"><code>encode</code></h3>

``` python
encode(instance)
```

Encode a tf.transform encoded dict as tf.Example.



