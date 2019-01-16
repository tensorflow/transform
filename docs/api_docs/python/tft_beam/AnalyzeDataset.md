<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft_beam.AnalyzeDataset" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="label"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__long__"/>
<meta itemprop="property" content="__native__"/>
<meta itemprop="property" content="__nonzero__"/>
<meta itemprop="property" content="__or__"/>
<meta itemprop="property" content="__ror__"/>
<meta itemprop="property" content="__rrshift__"/>
<meta itemprop="property" content="__unicode__"/>
<meta itemprop="property" content="default_label"/>
<meta itemprop="property" content="default_type_hints"/>
<meta itemprop="property" content="display_data"/>
<meta itemprop="property" content="expand"/>
<meta itemprop="property" content="from_runner_api"/>
<meta itemprop="property" content="get_type_hints"/>
<meta itemprop="property" content="get_windowing"/>
<meta itemprop="property" content="infer_output_type"/>
<meta itemprop="property" content="next"/>
<meta itemprop="property" content="register_urn"/>
<meta itemprop="property" content="runner_api_requires_keyed_input"/>
<meta itemprop="property" content="to_runner_api"/>
<meta itemprop="property" content="to_runner_api_parameter"/>
<meta itemprop="property" content="to_runner_api_pickled"/>
<meta itemprop="property" content="type_check_inputs"/>
<meta itemprop="property" content="type_check_inputs_or_outputs"/>
<meta itemprop="property" content="type_check_outputs"/>
<meta itemprop="property" content="with_input_types"/>
<meta itemprop="property" content="with_output_types"/>
<meta itemprop="property" content="pipeline"/>
<meta itemprop="property" content="side_inputs"/>
</div>

# tft_beam.AnalyzeDataset

## Class `AnalyzeDataset`



Takes a preprocessing_fn and computes the relevant statistics.

AnalyzeDataset accepts a preprocessing_fn in its constructor.  When its
`expand` method is called on a dataset, it computes all the relevant
statistics required to run the transformation described by the
preprocessing_fn, and returns a TransformFn representing the application of
the preprocessing_fn.

#### Args:

* <b>`preprocessing_fn`</b>: A function that accepts and returns a dictionary from
    strings to `Tensor` or `SparseTensor`s.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(preprocessing_fn)
```





## Properties

<h3 id="label"><code>label</code></h3>





## Methods

<h3 id="__long__"><code>__long__</code></h3>

``` python
__long__()
```



<h3 id="__native__"><code>__native__</code></h3>

``` python
__native__()
```

Hook for the future.utils.native() function

<h3 id="__nonzero__"><code>__nonzero__</code></h3>

``` python
__nonzero__()
```



<h3 id="__or__"><code>__or__</code></h3>

``` python
__or__(right)
```

Used to compose PTransforms, e.g., ptransform1 | ptransform2.

<h3 id="__ror__"><code>__ror__</code></h3>

``` python
__ror__(
    left,
    label=None
)
```

Used to apply this PTransform to non-PValues, e.g., a tuple.

<h3 id="__rrshift__"><code>__rrshift__</code></h3>

``` python
__rrshift__(label)
```



<h3 id="__unicode__"><code>__unicode__</code></h3>

``` python
__unicode__()
```



<h3 id="default_label"><code>default_label</code></h3>

``` python
default_label()
```



<h3 id="default_type_hints"><code>default_type_hints</code></h3>

``` python
default_type_hints()
```



<h3 id="display_data"><code>display_data</code></h3>

``` python
display_data()
```

Returns the display data associated to a pipeline component.

It should be reimplemented in pipeline components that wish to have
static display data.

#### Returns:

Dict[str, Any]: A dictionary containing ``key:value`` pairs.
The value might be an integer, float or string value; a
:class:`DisplayDataItem` for values that have more data
(e.g. short value, label, url); or a :class:`HasDisplayData` instance
that has more display data that should be picked up. For example::

  {
    'key1': 'string_value',
    'key2': 1234,
    'key3': 3.14159265,
    'key4': DisplayDataItem('apache.org', url='http://apache.org'),
    'key5': subComponent
  }

<h3 id="expand"><code>expand</code></h3>

``` python
expand(dataset)
```



<h3 id="from_runner_api"><code>from_runner_api</code></h3>

``` python
from_runner_api(
    cls,
    proto,
    context
)
```



<h3 id="get_type_hints"><code>get_type_hints</code></h3>

``` python
get_type_hints()
```



<h3 id="get_windowing"><code>get_windowing</code></h3>

``` python
get_windowing(inputs)
```

Returns the window function to be associated with transform's output.

By default most transforms just return the windowing function associated
with the input PCollection (or the first input if several).

<h3 id="infer_output_type"><code>infer_output_type</code></h3>

``` python
infer_output_type(unused_input_type)
```



<h3 id="next"><code>next</code></h3>

``` python
next()
```



<h3 id="register_urn"><code>register_urn</code></h3>

``` python
register_urn(
    cls,
    urn,
    parameter_type,
    constructor=None
)
```



<h3 id="runner_api_requires_keyed_input"><code>runner_api_requires_keyed_input</code></h3>

``` python
runner_api_requires_keyed_input()
```



<h3 id="to_runner_api"><code>to_runner_api</code></h3>

``` python
to_runner_api(
    context,
    has_parts=False
)
```



<h3 id="to_runner_api_parameter"><code>to_runner_api_parameter</code></h3>

``` python
to_runner_api_parameter(unused_context)
```



<h3 id="to_runner_api_pickled"><code>to_runner_api_pickled</code></h3>

``` python
to_runner_api_pickled(unused_context)
```



<h3 id="type_check_inputs"><code>type_check_inputs</code></h3>

``` python
type_check_inputs(pvalueish)
```



<h3 id="type_check_inputs_or_outputs"><code>type_check_inputs_or_outputs</code></h3>

``` python
type_check_inputs_or_outputs(
    pvalueish,
    input_or_output
)
```



<h3 id="type_check_outputs"><code>type_check_outputs</code></h3>

``` python
type_check_outputs(pvalueish)
```



<h3 id="with_input_types"><code>with_input_types</code></h3>

``` python
with_input_types(input_type_hint)
```

Annotates the input type of a :class:`PTransform` with a type-hint.

#### Args:

input_type_hint (type): An instance of an allowed built-in type, a custom
  class, or an instance of a
  :class:`~apache_beam.typehints.typehints.TypeConstraint`.


#### Raises:

~exceptions.TypeError: If **input_type_hint** is not a valid type-hint.
  See
  :obj:`apache_beam.typehints.typehints.validate_composite_type_param()`
  for further details.


#### Returns:

* <b>`PTransform`</b>: A reference to the instance of this particular
  :class:`PTransform` object. This allows chaining type-hinting related
  methods.

<h3 id="with_output_types"><code>with_output_types</code></h3>

``` python
with_output_types(type_hint)
```

Annotates the output type of a :class:`PTransform` with a type-hint.

#### Args:

type_hint (type): An instance of an allowed built-in type, a custom class,
  or a :class:`~apache_beam.typehints.typehints.TypeConstraint`.


#### Raises:

~exceptions.TypeError: If **type_hint** is not a valid type-hint. See
  :obj:`~apache_beam.typehints.typehints.validate_composite_type_param()`
  for further details.


#### Returns:

* <b>`PTransform`</b>: A reference to the instance of this particular
  :class:`PTransform` object. This allows chaining type-hinting related
  methods.



## Class Members

<h3 id="pipeline"><code>pipeline</code></h3>

<h3 id="side_inputs"><code>side_inputs</code></h3>

