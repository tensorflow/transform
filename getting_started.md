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
is a logical description of a transformation of a dataset.  The preprocessing
function accepts and returns a dictionary of tensors (in this guide, "tensors"
generally means `Tensor`s or `SparseTensor`s).  There are two kinds of functions
that can be used to define the preprocessing function:

1) Any function that accepts and returns tensors.  These will add TensorFlow
operations to the graph that transforms raw data into transformed data.

2) Any of the tf.Transform provided "analyzers". Analyzers also accept and return
tensors, but unlike typical TensorFlow functions they don't add TF Operations
to the graph.  Instead, they cause tf.Transform to compute a full pass operation
outside of TensorFlow, using the input tensor values over the full dataset to
generate a constant tensor that gets returned as the output.  For example
`tft.min` computes the minimum of a tensor over the whole dataset. Currently
tf.Transform provides a fixed set of analyzers, but this will be extensible in
future versions.

By combining analyzers and regular TensorFlow functions, users can flexibly
create pipelines for transforming their data.  The following preprocessing
function transforms each of three features in different ways, and combines two
of the features.

```
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

`x`, `y` and `s` are `Tensor`s that represent input features. The first new
tensor to be constructed, `x_centered`, is constructed by applying `tft.mean`
to `x` and subtracting this from `x`. `tft.mean(x)` returns a tensor
representing the mean of the tensor `x`. Thus `x_centered` is the tensor `x`
with the mean subtracted.

The second new tensor is `y_normalized`, created in a similar manner but using
the convenience method `tft.scale_to_0_1`. This method does something similar
under the hood to what is done to compute `x_centered`, namely computing a max
and min and using these to scale `y`.

The tensor `s_integerized` shows an example of string manipulation. In this
simple case we take a string and map it to an integer. This too uses a
convenience function, `tft.string_to_int`.   This function uses an analyzer to
compute the unique values taken by the input strings, and then uses TensorFlow
ops to convert the input strings to indices in the table of unique values.

The final column shows that it is possible to use tensorflow operations to
create new features by combining tensors.

The preprocessing function defines a pipeline of operations on a dataset.  In
order to apply such a pipeline, we rely on a concrete implementation of the
tf.Transform API. The Apache Beam implementation provides `PTransform`s that
apply a user's preprocessing function to data. The typical workflow of a
tf.Transform user will be to construct a preprocessing function, and then
incorporate this into a larger Beam pipeline, ultimately materializing the data
for training.

### A Note on Batching

Batching is an important part of TensorFlow. Since one of the goals of
tf.Transform is to provide the TensorFlow graph for preprocessing that can be
incorporated into the serving graph (and optionally the training graph),
batching is also an important concept in tf.Transform.

While it is not obvious from the example above, the user defined preprocessing
function will be passed tensors representing *batches*, not individual
instances, just as will happen during training and serving with TensorFlow.  On
the other hand, analyzers perform a computation over the whole dataset and
return a single value, not a batch of values.  Thus `x` is a `Tensor` of shape
`(batch_size,)` while `tft.mean(x)` is a `Tensor` of shape `()`.  The
subtraction `x - tft.mean(x)` involves broadcasting where the value of
`tft.mean(x)` is subtracted from every element of the batch represented by `x`.

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
transform_fn = (
    (raw_data, raw_data_metadata)
    | beam_impl.AnalyzeDataset(preprocessing_fn))
transformed_dataset = (
    ((raw_data, raw_data_metadata), transform_fn)
    | beam_impl.TransformDataset())
transformed_data, transformed_metadata = transformed_dataset
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
    's': dataset_schema.ColumnSchema(tf.string, [],
        dataset_schema.FixedColumnRepresentation()),
    'y': dataset_schema.ColumnSchema(tf.float32, [],
        dataset_schema.FixedColumnRepresentation()),
    'x': dataset_schema.ColumnSchema(tf.float32, [],
        dataset_schema.FixedColumnRepresentation())
}))

```

The `dataset_schema.Schema` class is a wrapper around a dict of
`dataset_schema.ColumnSchema`. Each key in the dict describes the logical name
of a tensor, and the `ColumnSchema` describes both the kind of tensor and how it
is represented in-memory or on-disk.

The first argument to `ColumnSchema` specifies the `Domain` which includes the
data type and richer infomation such as ranges. In our case we only specify the
data type and use a helper function to create the `Domain`. The second argument
provides a list of `Axis` objects describing the shape of the tensor. In our
example the shape has no axes because the values are scalars (rank 0 tensors).

The third argument to `ColumnSchema` is the representation of the data. There
are three kinds of representation. A `FixedColumnRepresentation` is a
representation of a column with fixed, known size. This allows each instance to
be represented as a list that can be packed into a tensor of that size. See
`tf_metadata/dataset_schema.py` for a description of the other kinds of
representation.

Note that while the shape of the tensor is determined by its axes, whether it is
represented by a `Tensor` or `SparseTensor` in the graph is determined by the
representation. This makes sense since data stored in a sparse format naturally
is mapped to a sparse tensor, and users are free to convert between `Tensor`s
and `SparseTensor`s in their custom code.

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

  # For the label column we provide the mapping from string to index.
  def convert_label(label):
    table = lookup.string_to_index_table_from_tensor(['>50K', '<=50K'])
    return table.lookup(label)
  outputs[LABEL_COLUMN] = tft.apply_function(
      convert_label, inputs[LABEL_COLUMN])

  return outputs
```

One difference from the previous example is that for the label column, we
manually specify the mapping from string to index so that ">50K" gets mapped to
0 and "<=50K" gets mapped to 1. This is useful so that we know which index in
the trained model corresponds to which label.  We cannot apply the function
`convert_label` directly to its arguments because `tf.Transform` needs to know
about the `Table` defined in `convert_label`.  That is, `convert_label` is not
a pure function but involves table initialization.  For such functions, we use
`tft.apply_function` to wrap the function application.

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

The next step is to create in input function builder that generates the input
function for training and eval.  The main difference from usual training using
tf.Learn, is that to parse the transformed data, we don't have to provide a
feature spec.  Instead, we take the metadata for the transformed data and use
it to generate a feature spec.

```
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

The rest of the code is the same as the usual use of the `Estimator` class, see
the [TensorFlow
documentation](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/BaseEstimator)
for more details.

The example code also contains code to export the model in the SavedModel
format.  The exported model can be used by
[Tensorflow Serving](https://www.tensorflow.org/serving/serving_basic) or
[Cloud ML Engine](https://cloud.google.com/ml-engine/docs/prediction-overview).
