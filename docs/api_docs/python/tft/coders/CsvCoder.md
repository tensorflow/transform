<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tft.coders.CsvCoder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="decode"/>
<meta itemprop="property" content="encode"/>
</div>

# tft.coders.CsvCoder

## Class `CsvCoder`



A coder to encode and decode CSV formatted data.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    column_names,
    schema,
    delimiter=',',
    secondary_delimiter=None,
    multivalent_columns=None
)
```

Initializes CsvCoder.

#### Args:

* <b>`column_names`</b>: Tuple of strings. Order must match the order in the file.
* <b>`schema`</b>: A `Schema` object.
* <b>`delimiter`</b>: A one-character string used to separate fields.
* <b>`secondary_delimiter`</b>: A one-character string used to separate values within
    the same field.
* <b>`multivalent_columns`</b>: A list of names for multivalent columns that need to
    be split based on secondary delimiter.


#### Raises:

* <b>`ValueError`</b>: If `schema` is invalid.



## Methods

<h3 id="decode"><code>decode</code></h3>

``` python
decode(csv_string)
```

Decodes the given string record according to the schema.

Missing value handling is as follows:

1. For FixedLenFeature:
    1. If FixedLenFeature and has a default value, use that value for
    missing entries.
    2. If FixedLenFeature and doesn't have default value throw an Exception
    on missing entries.

2. For VarLenFeature return an empty array.

3. For SparseFeature throw an Exception if only one of the indices or values
   has a missing entry. If both indices and values are missing, return
   a tuple of 2 empty arrays.

For the case of multivalent columns a ValueError will occur if
FixedLenFeature gets the wrong number of values, or a SparseFeature gets
different length indices and values.

#### Args:

* <b>`csv_string`</b>: String to be decoded.


#### Returns:

Dictionary of column name to value.


#### Raises:

* <b>`DecodeError`</b>: If columns do not match specified csv headers.
* <b>`ValueError`</b>: If some numeric column has non-numeric data, if a
      SparseFeature has missing indices but not values or vice versa or
      multivalent data has the wrong length.

<h3 id="encode"><code>encode</code></h3>

``` python
encode(instance)
```

Encode a tf.transform encoded dict to a csv-formatted string.

#### Args:

* <b>`instance`</b>: A python dictionary where the keys are the column names and the
    values are fixed len or var len encoded features.


#### Returns:

A csv-formatted string. The order of the columns is given by column_names.



