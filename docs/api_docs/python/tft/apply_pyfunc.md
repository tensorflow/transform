<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_pyfunc" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_pyfunc

``` python
tft.apply_pyfunc(
    func,
    Tout,
    stateful=True,
    name=None,
    *args
)
```

Applies a python function to some `Tensor`s.

Applies a python function to some `Tensor`s given by the argument list. The
number of arguments should match the number of inputs to the function.

This function is for using inside a preprocessing_fn.  It is a wrapper around
`tf.py_func`.  A function added this way can run in Transform, and during
training when the graph is imported using the `transform_raw_features` method
of the `TFTransformOutput` class.  However if the resulting training graph is
serialized and deserialized, then the `tf.py_func` op will not work and will
cause an error.  This means that TensorFlow Serving will not be able to serve
this graph.

The underlying reason for this limited support is that `tf.py_func` ops were
not designed to be serialized since they contain a reference to arbitrary
Python functions. This function pickles those functions and including them in
the graph, and `transform_raw_features` similarly unpickles the functions.
But unpickling requires a Python environment, so there it's not possible to
provide support in non-Python languages for loading such ops.  Therefore
loading these ops in libraries such as TensorFlow Serving is not supported.

#### Args:

* <b>`func`</b>: A Python function, which accepts a list of NumPy `ndarray` objects
    having element types that match the corresponding `tf.Tensor` objects
    in `*args`, and returns a list of `ndarray` objects (or a single
    `ndarray`) having element types that match the corresponding values
    in `Tout`.
* <b>`Tout`</b>: A list or tuple of tensorflow data types or a single tensorflow data
    type if there is only one, indicating what `func` returns.
* <b>`stateful`</b>: (Boolean.) If True, the function should be considered stateful.
    If a function is stateless, when given the same input it will return the
    same output and have no observable side effects. Optimizations such as
    common subexpression elimination are only performed on stateless
    operations.
* <b>`name`</b>: A name for the operation (optional).
* <b>`*args`</b>: The list of `Tensor`s to apply the arguments to.

#### Returns:

A `Tensor` representing the application of the function.