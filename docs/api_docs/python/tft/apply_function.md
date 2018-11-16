<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.apply_function" />
<meta itemprop="path" content="Stable" />
</div>

# tft.apply_function

``` python
tft.apply_function(
    fn,
    *args
)
```

Deprecated function, equivalent to fn(*args). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
apply_function is no longer needed.  `apply_function(fn, *args)` is now equvalent to `fn(*args)`

In previous versions of tf.Transform, it was necessary to wrap function
application in `apply_function`, that is call apply_function(fn, *args)
instead of calling fn(*args) directly.  This was necessary due to limitations
in the ability of tf.Transform to inspect the TensorFlow graph.  These
limitations no longer apply so apply_function is no longer needed.

#### Args:

* <b>`fn`</b>: The function to apply.
* <b>`*args`</b>: The arguments to apply `fn` to.


#### Returns:

The results of applying fn.