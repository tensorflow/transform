<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.sanitized_vocab_filename" />
<meta itemprop="path" content="Stable" />
</div>

# tft.sanitized_vocab_filename

``` python
tft.sanitized_vocab_filename(
    filename=None,
    prefix=None
)
```

Generates a sanitized filename either from the given filename or the scope.

If filename is specified, provide a sanitized version of the given filename.
Otherwise generate a filename from the current scope.  Note that it is the
callers responsibility to ensure that filenames are unique across calls within
a given preprocessing function.

#### Args:

* <b>`filename`</b>: A filename with non-alpha characters replaced with underscores and
    spaces to hyphens.
* <b>`prefix`</b>: Prefix to use for the name of the vocab file, if filename
    is not given.


#### Returns:

A valid filename.


#### Raises:

* <b>`ValueError`</b>: If neither filename and prefix are specified, or if both
    are specified.