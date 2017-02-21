# Getting Started with tf.Transform

This guide introduces the basic concepts of tf.Transform and how to use them
with some examples. We first describe how to define a "preprocessing function"
which is a logical description of the pipeline that transforms the raw data into
the data that will be used to train an ML model. We then describe how the Beam
implementation is used to actually transform data, by converting the user's
preprocessing function into a Beam pipeline. The subsequent sections cover other
aspects of the usage of tf.Transform.

## Defining a Preprocessing Function

The most important concept of tf.Transform is the "preprocessing function". This
is a logical description of a transformation of a dataset. The dataset is
conceptualized as a dictionary of columns, and the preprocessing function is
defined by two basic mechanisms:

1) Applying `tft.map`, which takes a user-defined function that accepts and
returns tensors. Such a function can use any TensorFlow operation to construct
the output tensors from the inputs. The remaining arguments of `tft.map` are the
columns that the function should be applied to. The number of columns provided
should equal the number of arguments to the user-defined function. Like the
Python `map` function, `tft.map` applies the user-provided function to the
elements in the columns specified. Each row is treated independently, and the
output is a column containing the results (but see the note on batching at the
end of this section).

2) Applying any of the tf.Transform provided "analyzers". Analyzers are
functions that accept one or more `Column`s and return some summary statistic
for the input column or columns. A statistic is like a column except that it
only has a single value. An example of an analyzer is `tft.min` which computes
the minimum of a column. Currently tf.Transform provides a fixed set of
analyzers, but this will be extensible in future versions.

In fact, `tft.map` can also accept statistics, which is how statistics are
incorporated into the user-defined pipeline. By combining analyzers and
`tft.map`, users can flexibly create pipelines for transforming their data. In
particular, users should define a "preprocessing function" which accepts and
returns columns.

The following preprocessing function transforms each of three columns in
different ways, and combines two of the columns.

```
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = tft.map(lambda x, mean: x - mean, x, tft.mean(x))
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.string_to_int(s)
  x_centered_times_y_normalized = tft.map(lambda x, y: x * y,
                                          x_centered, y_normalized)
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }
```

`x`, `y` and `s` are local variables that represent input columns, that are
declared for code brevity. The first new column to be constructed, `x_centered`,
is constructed by composing `tft.map` and `tft.mean`. `tft.mean(x)` returns a
statistic representing the mean of the column `x`. The lambda passed to
`tft.map` is simply subtraction, where the first argument is the column `x` and
the second is the statistic `tft.mean(x)`. Thus `x_centered` is the column `x`
with the mean subtracted.

The second new column is `y_normalized`, created in a similar manner but using
the convenience method `tft.scale_to_0_1`. This method does something similar
under the hood to what is done to compute `x_centered`, namely computing a max
and min and using these to scale `y`.

The column `s_integerized` shows an example of string manipulation. In this
simple case we take a string and map it to an integer. This too uses a
convenience function, where the analyzer that is applied computes the unique
values taken by the column, and the map uses these values as a dictionary to
convert to an integer.

The final column shows that it is possible to use `tft.map` not only to
manipulate a single column but also to combine columns.

Note that `Column`s are not themselves wrappers around data. Rather they are
placeholders used to construct a definition of the user's logical pipeline. In
order to apply such a pipeline to data, we rely on a concrete implementation of
the tf.Transform API. The Apache Beam implementation provides `PTransform`s that
apply a user's preprocessing function to data. The typical workflow of a
tf.Transform user will be to construct a preprocessing function, and then
incorporate this into a larger Beam pipeline, ultimately materializing the data
for training.

### A Note on Batching

Batching is an important part of TensorFlow. Since one of the goals of
tf.Transform is to provide the TensorFlow graph for preprocessing that can be
incorporated into the serving graph (and optionally the training graph),
batching is also an important concept in tf.Transform.

While it is not obvious from the example above, the user defined function passed
to `tft.map` will be passed tensors representing *batches*, not individual
instances, just as will happen during training and serving with TensorFlow. This
is only the case for inputs that are `Column`s, not `Statistic`s. Thus the
actual tensors used in the `tft.map` for `x_centered` are 1) a rank 1 tensor,
representing a batch of values from the column `x`, whose first dimension is the
batch dimension; and 2) a rank 0 tensor representing the mean of that column.

## The Canonical Beam Implementation

While the preprocessing function is intended as a logical description of a
preprocessing pipeline that can be implemented on a variety of data processing
frameworks, tf.Transform provides a canonical implementation that runs the
preprocessing function on Apache Beam. This implementation also demonstrates the
kind of functionality that is required from an implementation. There is no
formal API for this functionality, so that each implementation can use an API
that is idiomatic for its particular data processing framework.

The Beam implementation provides two `PTransform`s that are used to process data
given a preprocessing function. We begin with the composite `PTransform`
`AnalyzeAndTransformDataset`. Sample usage is shown below.

```
raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

raw_data_metadata = ...
transformed_dataset, transform_fn = (
    (raw_data, raw_data_metadata) | beam_impl.AnalyzeAndTransformDataset(
        preprocessing_fn, tempfile.mkdtemp()))
```

The content of `transformed_data` is shown below, and can be seen to contain the
transformed columns in the same format as the raw data. In particular, the
values of `s_integerized` are `[0, 1, 0]` (these values depend on how the words
`hello` and `world` were mapped to integers, which is deterministic). For the
column `x_centered` we subtracted the mean, so the values of the column `x`,
which were `[1.0, 2.0, 3.0]` became `[-1.0, 0.0, 1.0]`. Similarly the rest of
the columns match their expected values.

```
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

Both `raw_data` and `transformed_data` are datasets. See the next two sections
for how the Beam implementation represents datasets, and how to read/write data
from disk. The other return value, `transform_fn`, is a representation of the
transformation that was done to the data, which we discuss in more detail below.

In fact, `AnalyzeAndTransformDataset` is the composition of the two fundamental
transforms provided by the implementation, `AnalyzeDataset` and
`TransformDataset`. That is, the two code snippets below are equivalent.

```
transformed_data, transform_fn = (
    my_data | AnalyzeAndTransformDataset(preprocessing_fn))
```

```
transform_fn = my_data | AnalyzeDataset(preprocessing_fn)
transformed_data = (my_data, transform_fn) | TransformDataset()
```

The `transform_fn` is a pure function that represents an operation that is
applied to each row of the dataset. In particular, all the analyzer values are
already computed and treated as constants. In our example, the `transform_fn`
would contain as constants the mean of column `x`, min and max of column `y` and
the vocabulary used to map the strings to integers.

A key feature of tf.Transform is that `transform_fn` represents a map over rows,
that is it is a pure function that is applied to each row separately. All of the
computation involving aggregating over rows is done in `AnalyzeDataset`.
Furthermore, the `transform_fn` is representable as a TensorFlow `Graph` which
means that it can be embedded into the serving graph.

We provide `AnalyzeAndTransformDataset` in order to allow for optimizations that
are possible in this special case. This is exactly the same pattern as is used
in [scikit-learn](http://scikit-learn.org/stable/index.html), which provides the
`fit`, `transform` and `fit_transform` methods for preprocessors.

## Data Formats and Schema

In the code samples in the previous section we omitted the code that defined
`raw_data_metadata`. The metadata contains the schema that defines the layout of
the data so that it can be read from and written to various formats, as
discussed below. Even the in-memory format shown in the last section is not
self-describing and requires the schema in order to be interpreted as tensors.

Below we show the definition of the schema for the example data.

```
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
    'x': dataset_schema.ColumnSchema(
        dataset_schema.LogicalColumnSchema(
            dataset_schema.Domain(tf.float32), dataset_schema.LogicalShape([])),
        dataset_schema.FixedColumnRepresentation()),
    'y': dataset_schema.ColumnSchema(
        dataset_schema.LogicalColumnSchema(
            dataset_schema.Domain(tf.float32), dataset_schema.LogicalShape([])),
        dataset_schema.FixedColumnRepresentation()),
    's': dataset_schema.ColumnSchema(
        dataset_schema.LogicalColumnSchema(
            dataset_schema.Domain(tf.string), dataset_schema.LogicalShape([])),
        dataset_schema.FixedColumnRepresentation())
})
```

The `dataset_schema.Schema` class is a wrapper around a dict of
`dataset_schema.ColumnSchema`. Each key in the dict describes the logical name
of a tensor, and the `ColumnSchema` describes both the kind of tensor and how it
is represented in-memory or on-disk.

The `LogicalColumnSchema` defines the data type (and optionally, richer
information such as ranges) and the shape of the tensor. In our example the
shape has no axes because the values are scalars (rank 0 tensors). In general,
the shape is specified by a tuple with elements of type `Axis`, each of which
provides the size of each dimension. The `Axis` class may later support richer
information such as axis names.

The second part of the `ColumnSchema` is the representation of the data. There
are three kinds of representation. A `FixedColumnRepresentation` is a
representation of a column with fixed, known size. This allows each instance to
be represented as a list that can be packed into a tensor of that size. See
`tf_metadata/dataset_schema.py` for a description of the other kinds of
representation.

Note that while the shape of the tensor is determined by its logical schema,
whether it is represented by a `Tensor` or `SparseTensor` in the graph is
determined by the representation. This makes sense since data stored in a sparse
format naturally is mapped to a sparse tensor, and users are free to convert
between `Tensor`s and `SparseTensor`s in their custom code.

## IO with the Beam Implementation

So far we have worked with lists of dictionaries as the data format. This is a
simplification that relies on Beam's ability to work with lists as well as its
main representation of data, `PCollection`s. A `PCollection` is a representation
of data that forms a part of a Beam pipeline. A Beam pipeline can be formed by
applying various `PTransforms`, including `AnalyzeDataset` and
`TransformDataset`, and then running the pipeline. `PCollections` are not
materialized in the memory of the main binary, but instead are distributed among
the workers (although in this section we use the in-memory execution mode).

In this section, we start a new example that involves both reading and writing
data on disk, and representing data as a `PCollection` (not a list). We will
follow the code sample `census_example.py`. See the end of this section on how
to run an download the data and run this example. The "Census Income" dataset is
provided by the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). This dataset
contains both categorical and numeric data.

The raw data is in CSV format. Below, the first two lines from the data are
shown.

```
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
```

The columns of the dataset (see
[source](https://archive.ics.uci.edu/ml/datasets/Census+Income) for more
information), are either categorical or numeric. Since there are a large number
of columns, we generate a `Schema` similar to the last example, but do so by
looping through all columns of each type. See the sample code for more details.
Note that this dataset describes a classification problem, predicting the last
column which is whether the individual earns more or less than 50K per year.
However, from the point of view of tf.Transform, the label is just another
categorical column.

Having constructed the schema, we can use this schema to read the data from the
CSV file. `ordered_columns` is a constant that contains the list of all columns
in the order they appear in the CSV file and is needed since the schema does not
contain this information. We have excluded some extra beam transforms that we do
between reading the lines of the CSV file, and applying the converter that
converts each CSV row to an instance in the in-memory format.

```
converter = csv_coder.CsvCoder(ordered_columns, raw_data_schema)

raw_data = (
    p
    | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
    | ...
    | 'DecodeTrainData' >> beam.Map(converter.decode))
```

Preprocessing then proceeds similarly to the previous example, except that we
programmatically generate the preprocessing function instead of manually
specifying each column. The preprocessing function is shown below.
`NUMERICAL_COLUMNS` and `CATEGORICAL_COLUMNS` are lists that contain the names
of the numeric and categorical columns respectively.

```
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

  # Update outputs of both kinds to convert from shape (batch,), i.e. a batch
  # of scalars, to shape (batch, 1), i.e. a batch of vectors of length 1.
  # This is needed so the output can be easily wrapped in `FeatureColumn`s.
  for key in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
    outputs[key] = tft.map(lambda x: tf.expand_dims(x, -1), outputs[key])

  # For the label column we provide the mapping from string to index.
  def convert_label(label):
    table = lookup.string_to_index_table_from_tensor(['>50K', '<=50K'])
    return table.lookup(label)
  outputs[LABEL_COLUMN] = tft.map(convert_label, inputs[LABEL_COLUMN])

  return outputs
```

One difference from the previous example is that we convert the outputs from
scalars to single element vectors. This allows the data to be correctly read
during training. Also for the label column, we manually specify the mapping from
string to index so that ">50K" gets mapped to 0 and "<=50K" gets mapped to 1.
This is useful so that we know which index in the trained model corresponds to
which label.

The `raw_data` variable represents a `PCollection` containing data in the same
format as the list `raw_data` from the previous example, and the use of the
`AnalyzeAndTransformDataset` transform is the same. Note that the schema is used
in two places: reading the data from the CSV file, and as an input to
`AnalyzeAndTransformDataset`. This is because both the CSV format and the
in-memory format need to be paired with a schema in order to interpret them as
tensors.

The final stage is to write the transformed data to disk, which has a similar
form to the reading of the raw data. The schema used to do this is part of the
output of `AnalyzeAndTransformDataset`. That is `AnalyzeAndTransformDataset`
infers a schema for the output data. The code to write to disk is shown below.
Note that the schema is a part of the metadata, but we can use the two
interchangeably within the tf.Transform API, e.g. pass the metadata to the
`ExampleProtoCoder`. Note also that we are writing to a different format.
Instead of `textio.WriteToText`, we use Beam's builtin support for the TFRecord
format, and use a coder to encode the data as `Example` protos. This is a better
format for use in training, as shown in the next section.
`transformed_eval_data_base` provides the base filename for the individual
shards that are written.

```
transformed_data | "WriteTrainData" >> tfrecordio.WriteToTFRecord(
    transformed_eval_data_base,
    coder=example_proto_coder.ExampleProtoCoder(transformed_metadata))
```

In addition to the training data, we also write out the metadata.

```
transformed_metadata | 'WriteMetadata' >> beam_metadata_io.WriteMetadata(
    transformed_metadata_file, pipeline=p)
```

The entire Beam pipeline is run by calling `p.run().wait_until_finish()`. Up
until this point, the Beam pipeline represents a deferred, distributed
computation; it provides instructions as to what is to be done, but these have
not yet been executed. This final call executes the specified pipeline.

### Downloading the Census dataset

The following shell commands can be used to download the census dataset.

```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

When running `census_example.py`, the directory containing this data should be
passed as the first argument. The script will create a temp subdirectory of the
data directory to put preprocessed data in.

## Integration with TensorFlow Training

The final section of `census_example.py` demonstrates how the preprocessed data
is used to train a model. See the [tf.Learn
documentation](https://www.tensorflow.org/get_started/tflearn) for more details.
The first step is to construct an `Estimator`. This requires a description of
the preprocessed columns. Each numeric column is described as a
`real_valued_column` which is a wrapper around a dense vector of fixed size (in
this case 1). Each categorical column is described as a
`sparse_column_with_integerized_feature`. This is a way of indicating that the
mapping from string to integers has already been done. We must provide the
bucket size, i.e. the max index contained in the column. For the Census data we
know these values already, but in general it would be good to have tf.Transform
compute them. Future versions of tf.Transform will write this information out as
part of the metadata which can then be used here.

```
real_valued_columns = [feature_column.real_valued_column(key)
                       for key in NUMERIC_COLUMNS]

one_hot_columns = [
    feature_column.sparse_column_with_integerized_feature(
        key, bucket_size=bucket_size)
    for key, bucket_size in zip(CATEGORICAL_COLUMNS, BUCKET_SIZES)]

estimator = learn.LinearClassifier(real_valued_columns + one_hot_columns)
```

The next step is to create in input function that reads the data. Since the data
was created with tf.Transform, it can be read in using the metadata for the
transformed data. The input function is created with `build_training_input_fn`
which accepts the metadata, the location of the transformed data, the batch size
and the column of the data that contains the training label.

```
transformed_metadata = metadata_io.read_metadata(transformed_metadata_file)
train_input_fn = input_fn_maker.build_training_input_fn(
    transformed_metadata,
    PREPROCESSED_TRAIN_DATA + '*',
    training_batch_size=TRAIN_BATCH_SIZE,
    label_keys=['label'])
```

The rest of the code is the same as the usual use of the `Estimator` class, see
the [TensorFlow
documentation](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/BaseEstimator)
for more details.
