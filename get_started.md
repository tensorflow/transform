<!-- See: www.tensorflow.org/tfx/transform/ -->

# Get Started with TensorFlow Transform

This guide introduces the basic concepts of `tf.Transform` and how to use
them. It will:

* Define a *preprocessing function*, a logical description of the pipeline that
  transforms the raw data into the data used to train a machine learning model.
* Show the [Apache Beam](https://beam.apache.org/) implementation used to
  transform data by converting the *preprocessing function* into a
  *Beam pipeline*.
* Show additional usage examples.

## Define a preprocessing function

The *preprocessing function* is the most important concept of `tf.Transform`.
The preprocessing function is a logical description of a transformation of the
dataset. The preprocessing function accepts and returns a dictionary of tensors,
where a *tensor* means `Tensor` or `SparseTensor`. There are two kinds of
functions used to define the preprocessing function:

1. Any function that accepts and returns tensors. These add TensorFlow operations
   to the graph that transform raw data into transformed data.
2. Any of the *analyzers* provided by `tf.Transform`. Analyzers also accept
   and return tensors, but unlike TensorFlow functions, they *do not* add
   operations to the graph. Instead, analyzers cause `tf.Transform` to compute
   a full-pass operation outside of TensorFlow. They use the input tensor values
   over the entire dataset to generate a constant tensor that is returned as the
   output. For example, `tft.min` computes the minimum of a tensor over the
   dataset. `tf.Transform` provides a fixed set of analyzers, but this will be
   extended in future versions.

### Preprocessing function example

By combining analyzers and regular TensorFlow functions, users can create
flexible pipelines for transforming data. The following preprocessing function
transforms each of the three features in different ways, and combines two of the
features:

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = x - tft.mean(x)
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.string_to_int(s)
  x_centered_times_y_normalized = x_centered * y_normalized
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }
```

Here, `x`, `y` and `s` are `Tensor`s that represent input features. The first
new tensor that is created, `x_centered`, is built by applying `tft.mean` to
`x` and subtracting this from `x`. `tft.mean(x)` returns a tensor representing
the mean of the tensor `x`. `x_centered` is the tensor `x` with the mean
subtracted.

The second new tensor, `y_normalized`, is created in a similar manner but using
the convenience method `tft.scale_to_0_1`. This method does something similar
to computing `x_centered`, namely computing a maximum and minimum and using
these to scale `y`.

The tensor `s_integerized` shows an example of string manipulation. In this
case, we take a string and map it to an integer. This uses the convenience
function `tft.string_to_int`. This function uses an analyzer to compute the
unique values taken by the input strings, and then uses TensorFlow operations to
convert the input strings to indices in the table of unique values.

The final column shows that it is possible to use TensorFlow operations to create
new features by combining tensors.

The preprocessing function defines a pipeline of operations on a dataset. In
order to apply the pipeline, we rely on a concrete implementation of the
`tf.Transform` API. The Apache Beam implementation provides `PTransform`
which applies a user's preprocessing function to data. The typical workflow of a
`tf.Transform` user will construct a preprocessing function, then incorporate
this into a larger Beam pipeline, creating the data for training.

### Batching

Batching is an important part of TensorFlow. Since one of the goals of
`tf.Transform` is to provide a TensorFlow graph for preprocessing that can be
incorporated into the serving graph (and, optionally, the training graph),
batching is also an important concept in `tf.Transform`.

While not obvious in the example above, the user defined preprocessing function
is passed tensors representing *batches* and not individual instances, as happens
during training and serving with TensorFlow. On the other hand, analyzers
perform a computation over the entire dataset that returns a single value and not
a batch of values. `x` is a `Tensor` with a shape of `(batch_size,)`, while
`tft.mean(x)` is a `Tensor` with a shape of `()`. The subtraction
`x - tft.mean(x)` broadcasts where the value of `tft.mean(x)` is
subtracted from every element of the batch represented by `x`.

## Apache Beam Implementation

While the *preprocessing function* is intended as a logical description of a
*preprocessing pipeline* implemented on multiple data processing frameworks,
`tf.Transform` provides a canonical implementation used on Apache Beam. This
implementation demonstrates the functionality required from an implementation.
There is no formal API for this functionality, so each implementation can use an
API that is idiomatic for its particular data processing framework.

The Apache Beam implementation provides two `PTransform`s used to process data
for a preprocessing function. The following shows the usage for the composite
`PTransform AnalyzeAndTransformDataset`:

```python
raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

raw_data_metadata = ...
transformed_dataset, transform_fn = (
    (raw_data, raw_data_metadata) | beam_impl.AnalyzeAndTransformDataset(
        preprocessing_fn))
transformed_data, transformed_metadata = transformed_dataset
```

The `transformed_data` content is shown below and contains the transformed
columns in the same format as the raw data. In particular, the values of
`s_integerized` are `[0, 1, 0]`—these values depend on how the words
`hello` and `world` were mapped to integers, which is deterministic. For the
column `x_centered`, we subtracted the mean so the values of the column `x`,
which were `[1.0, 2.0, 3.0]`, became `[-1.0, 0.0, 1.0]`. Similarly, the
rest of the columns match their expected values.

```python
[{u's_integerized': 0,
  u'x_centered': -1.0,
  u'x_centered_times_y_normalized': -0.0,
  u'y_normalized': 0.0},
 {u's_integerized': 1,
  u'x_centered': 0.0,
  u'x_centered_times_y_normalized': 0.0,
  u'y_normalized': 0.5},
 {u's_integerized': 0,
  u'x_centered': 1.0,
  u'x_centered_times_y_normalized': 1.0,
  u'y_normalized': 1.0}]
```

Both `raw_data` and `transformed_data` are datasets. The next two sections
show how the Beam implementation represents datasets and how to read and write
data to disk. The other return value, `transform_fn`, represents the
transformation applied to the data, covered in detail below.

The `AnalyzeAndTransformDataset` is the composition of the two fundamental
transforms provided by the implementation `AnalyzeDataset` and
`TransformDataset`. So the following two code snippets are equivalent:

```python
transformed_data, transform_fn = (
    my_data | AnalyzeAndTransformDataset(preprocessing_fn))
```

```python
transform_fn = my_data | AnalyzeDataset(preprocessing_fn)
transformed_data = (my_data, transform_fn) | TransformDataset()
```

`transform_fn` is a pure function that represents an operation that is applied
to each row of the dataset. In particular, the analyzer values are already
computed and treated as constants. In the example, the `transform_fn` contains
as constants the mean of column `x`, the min and max of column `y`, and the
vocabulary used to map the strings to integers.

An important feature of `tf.Transform` is that `transform_fn` represents a
map *over rows*—it is a pure function applied to each row separately. All of the
computation for aggregating rows is done in `AnalyzeDataset`. Furthermore, the
`transform_fn` is represented as a TensorFlow `Graph` which can be embedded
into the serving graph.

`AnalyzeAndTransformDataset` is provided for optimizations in this special
case. This is the same pattern used in
[scikit-learn](http://scikit-learn.org/stable/index.html), providing the `fit`,
`transform`, and `fit_transform` methods.

## Data Formats and Schema

In the previous code examples, the code defining `raw_data_metadata` is
omitted. The metadata contains the schema that defines the layout of the data so
it is read from and written to various formats. Even the in-memory format shown
in the last section is not self-describing and requires the schema in order to be
interpreted as tensors.

Here's the definition of the schema for the example data:

```python
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
    's': dataset_schema.ColumnSchema(tf.string, [],
        dataset_schema.FixedColumnRepresentation()),
    'y': dataset_schema.ColumnSchema(tf.float32, [],
        dataset_schema.FixedColumnRepresentation()),
    'x': dataset_schema.ColumnSchema(tf.float32, [],
        dataset_schema.FixedColumnRepresentation())
}))

```

The `dataset_schema.Schema` class is a wrapper around a `dict` that contains
`dataset_schema.ColumnSchema`s. Each key describes the logical name of a
tensor and the `ColumnSchema` describes both the kind of tensor and how it
is represented in-memory or on-disk.

The first argument to `ColumnSchema` specifies the `Domain` that includes the
data type and more detailed information such as ranges. Here, only the data type
is specified and a helper function is used to create the `Domain`. The second
argument provides a list of `Axis` objects describing the shape of the tensor.
In this example, the shape has no axes because the values are scalars (rank 0
tensors).

The third argument to `ColumnSchema` is the representation of the data. There
are three kinds of representation. A `FixedColumnRepresentation` is a
representation of a column with fixed, known size. This allows each instance to
be represented as a list that can be packed into a tensor of that size. See
`tf_metadata/dataset_schema.py` for a description of the other kinds of
representation.

While the shape of the tensor is determined by its axes, if it's represented by a
`Tensor` or `SparseTensor` in the graph is determined by the representation.
Data stored in a sparse format is mapped to a sparse tensor, and users are free
to convert between `Tensor`s and `SparseTensor`s in their custom code.

## Input and output with Apache Beam

So far, the data format for the examples used lists of dictionaries. This is a
simplification that relies on Apache Beam's ability to work with lists as well as
its main representation of data, the `PCollection`. A `PCollection` is a
data representation that forms a part of a Beam pipeline. A Beam pipeline is
formed by applying various `PTransform`s, including `AnalyzeDataset` and
`TransformDataset`, and running the pipeline. A `PCollection` is not
created in the memory of the main binary, but instead is distributed among the
workers (although this section uses the in-memory execution mode).

The following example requires both reading and writing data on disk and
representing data as a `PCollection` (not a list), see:
[`census_example.py`](./examples/census_example.py).
Below we show how to download the data and run this example. The "Census Income"
dataset is provided by the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income).
This dataset contains both categorical and numeric data.

The data is in CSV format, here are the first two lines:

```
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
```

The columns of the dataset are either categorical or numeric. Since there are
many columns, a `Schema` is generated (similar to the previous example) by
looping through all columns of each type. This dataset describes a classification
problem: predicting the last column where the individual earns more or less than
50K per year. However, from the perspective of `tf.Transform`, this label is
just another categorical column.

Use this schema to read the data from the CSV file. The `ordered_columns`
constant contains the list of all columns in the order they appear in the CSV
file—required because the schema does not contain this information. Some extra
Beam transforms are removed since they're already done when reading from the CSV
file. Each CSV row is converted to an instance in the in-memory format.

```python
converter = tft.coders.CsvCoder(ordered_columns, raw_data_schema)

raw_data = (
    p
    | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
    | ...
    | 'DecodeTrainData' >> beam.Map(converter.decode))
```

Preprocessing is similar to the previous example, except the preprocessing
function is programmatically generated instead of manually specifying each
column. In the preprocessing function below, `NUMERICAL_COLUMNS` and
`CATEGORICAL_COLUMNS` are lists that contain the names of the numeric and
categorical columns:

```python
def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  outputs = {}

  # Scale numeric columns to have range [0, 1].
  for key in NUMERIC_COLUMNS:
    outputs[key] = tft.scale_to_0_1(inputs[key])

  # For all categorical columns except the label column, we use
  # tft.string_to_int which computes the set of unique values and uses this
  # to convert the strings to indices.
  for key in CATEGORICAL_COLUMNS:
    outputs[key] = tft.string_to_int(inputs[key])

  # For the label column we provide the mapping from string to index.
  def convert_label(label):
    table = lookup.string_to_index_table_from_tensor(['>50K', '<=50K'])
    return table.lookup(label)
  outputs[LABEL_COLUMN] = tft.apply_function(
      convert_label, inputs[LABEL_COLUMN])

  return outputs
```

One difference from the previous example is the label column manually specifies
the mapping from the string to an index. So `'>50'` is mapped to `0` and
`'<=50K'` is mapped to `1` because it's useful to know which index in the
trained model corresponds to which label. The `convert_label` function is not
applied directly to its arguments because `tf.Transform` needs to know about
the `Table` defined in `convert_label`. That is, `convert_label` is not a
pure function but does table initialization. For these functions, use
`tft.apply_function` to wrap the function application.

The `raw_data` variable represents a `PCollection` that contains data in the
same format as the list `raw_data` (from the previous example), using the same
`AnalyzeAndTransformDataset` transform. The schema is used in two places:
reading the data from the CSV file and as input to
`AnalyzeAndTransformDataset`. Both the CSV format and the in-memory format
must be paired with a schema in order to interpret them as tensors.

The final stage is to write the transformed data to disk and has a similar form
to reading the raw data. The schema used to do this is part of the output of
`AnalyzeAndTransformDataset` which infers a schema for the output data. The
code to write to disk is shown below. The schema is a part of the metadata but
uses the two interchangeably in the `tf.Transform` API (i.e. pass the metadata
to the `ExampleProtoCoder`). Be aware that this writes to a different format.
Instead of `textio.WriteToText`, use Beam's built-in support for the
`TFRecord` format and use a coder to encode the data as `Example` protos. This
is a better format to use for training, as shown in the next section.
`transformed_eval_data_base` provides the base filename for the individual
shards that are written.

```python
transformed_data | "WriteTrainData" >> tfrecordio.WriteToTFRecord(
    transformed_eval_data_base,
    coder=tft.coders.ExampleProtoCoder(transformed_metadata))
```

In addition to the training data, `transform_fn` is also written out with the
metadata:

```python
_ = (
    transform_fn
    | 'WriteTransformFn' >>
    transform_fn_io.WriteTransformFn(working_dir))
transformed_metadata | 'WriteMetadata' >> beam_metadata_io.WriteMetadata(
    transformed_metadata_file, pipeline=p)
```

Run the entire Beam pipeline with `p.run().wait_until_finish()`. Up until
this point, the Beam pipeline represents a deferred, distributed computation. It
provides instructions for what will be done, but the instructions have not been
executed. This final call executes the specified pipeline.

### Download the census dataset

Download the census dataset using the following shell commands:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data</code>
  <code class="devsite-terminal">wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test</code>
</pre>

When running the
[`census_example.py`](./examples/census_example.py) script, pass the
directory containing this data as the first argument. The script creates a
temporary sub-directory to add the preprocessed data.

## Integrate with TensorFlow Training

The final section of
[`census_example.py`](./examples/census_example.py) show how the preprocessed
data is used to train a model. See the
[premade Estimators documentation](http://tensorflow.org/get_started/premade_estimators)
for details. The first step is to construct an `Estimator` which requires a
description of the preprocessed columns. Each numeric column is described as a
`real_valued_column` that is a wrapper around a dense vector with a fixed
size (`1` in this example). Each categorical column is described as a
`sparse_column_with_integerized_feature`. This indicates the mapping
from string to integers has already been done. Provide the bucket size which is
the max index contained in the column. We already know the values for the census
data, but it's preferred to compute them using `tf.Transform`. Future versions
of `tf.Transform` will write this information out as part of the metadata that
can then be used here.

```python
real_valued_columns = [feature_column.real_valued_column(key)
                       for key in NUMERIC_COLUMNS]

one_hot_columns = [
    feature_column.sparse_column_with_integerized_feature(
        key, bucket_size=bucket_size)
    for key, bucket_size in zip(CATEGORICAL_COLUMNS, BUCKET_SIZES)]

estimator = learn.LinearClassifier(real_valued_columns + one_hot_columns)
```

The next step is to create a builder to generate the input function for training
and evaluation. The differs from the training used by `tf.Learn` since a
feature spec is not required to parse the transformed data. Instead, use the
metadata for the transformed data to generate a feature spec.

```python
def _make_training_input_fn(working_dir, filebase, batch_size):
  ...
  transformed_metadata = metadata_io.read_metadata(
      os.path.join(
          working_dir, transform_fn_io.TRANSFORMED_METADATA_DIR))
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  def input_fn():
    """Input function for training and eval."""
    transformed_features = tf.contrib.learn.io.read_batch_features(
        ..., transformed_feature_spec, ...)

    ...

  return input_fn
```

The remaining code is the same as using the
[`Estimator`](/api_docs/python/tf/contrib/learn/BaseEstimator) class. The
example also contains code to export the model in the `SavedModel` format. The
exported model can be used by
[Tensorflow Serving](https://www.tensorflow.org/serving/serving_basic) or the
[Cloud ML Engine](https://cloud.google.com/ml-engine/docs/prediction-overview).
