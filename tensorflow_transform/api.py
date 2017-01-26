"""The core public API of TFTransform.  Provide functions to transform columns.

The core TFTransform API provides a way for the user to construct an abstract
representation of a transformation from an input data set to an output data set.
This function is constructed using the provided functions that operate on the
`Column` and `Statistic` classes.  In particular, the user combines these
functions to build a function that accepts and returns a dictionary whose
keys are strings and whose values are `Column`s or `Statistics`.

This user-defined function then must be run using an implementation which can be
implemented with a number of backends, e.g. Apache Beam.  How the dataset
is represented, when deferred computation is done, and what sources and sinks
are available are all defined by the implementation.  The implementation must
subclass the AnalyzeAndTransformDataset, AnalyzeDataset and TransformDataset
classes, in a way that is API compatible with the canonical beam implementation
of these classes.  By API compatible we mean that it has equivalent objects
representing datasets and metadata.

Users should be able to write code such as

def preprocessing_fn(inputs):
  ...

with beam.Pipeline(...) as p:
  input = p | beam_impl.read_examples(..., schema)
  transformed, transform_fn = input | beam_impl.AnalyzeAndTransformDataset(
      preprocessing_fn)
  transformed | beam_impl.write_examples_and_metadata(
      examples_path, metadata_path)
  transform_fn | beam_impl.write_transform_fn(transform_fn_path)

The implementation should internally have types that we refer to as Dataset
and TransformFn (where the latter is a representation of the preprocessing
function).  The Dataset contains both the actual data and also metadata such as
the schema and various summary statistics for columns.

The implementations of AnalyzeAndTransformDataset etc. should be classes
providing an `expand(dataset)` method.  Any additional information needed to
perform the computation, e.g. the preprocessing_fn, are provided as constructor
arguments when instantiating these functions.  Thus the signatures are as
follows:

class AnalyzeDataset():
  def __init__(preprocessing_fn): ...
  def expand(dataset): ...  # returns TransformFn

class TransformDataset():
  def __init__(): ...
  def expand(dataset_and_transform_fn): ...  # returns Dataset

class AnalyzeAndTransformDataset():
  def __init(preprocessing_fn): ...
  def expand(dataset): ...  # returns (Dataset, TransformFn)

TODO(abrao): Investigate refactoring into smaller modules, e.g. analyzers,
transforms, core.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf


class Statistic(object):
  """Statistic represents a statistic of a column in a preprocessing function.

  The result of a summary statistic (e.g. mean, sum or a vocabulary) computed
  on one or more (possibly transformed) columns.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the statistic.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, tensor):
    self._tensor = tensor

  @property
  def tensor(self):
    return self._tensor


class Column(object):
  """A Column represents a column in a preprocessing function.

  Columns are either the columns of the input dataset or a column constructed
  by applying some row-wise transformation to the input dataset.

  Args:
    tensor: A `Tensor` or `SparseTensor` that will represent the column.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, tensor):
    self._tensor = tensor
    self._metadata_dict = {}

  @property
  def tensor(self):
    return self._tensor

  def add_metadata(self, key, value):
    """Sets the value of some metadata for this `Column`.

    TODO(kestert): Make this API compatible with the actual metadata format.

    Args:
      key: The metadata key
      value: Either a `Statistic` or primitive value or the appropriate type.

    Raises:
      KeyError: If `key` already exists in the metadata dict.
    """
    if key in self._metadata_dict:
      raise KeyError('Key %s already exists' % key)
    self._metadata_dict[key] = value


class _AnalyzerOutput(Statistic):
  """A Column containing the output of a transformation.

  A `_AnalyzerOutput` is defined by zero or more inputs (which may be `Column`s
  or `Statistic`s) and an analyzer applied to them.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the statistic.
    analyzer_name: The name of the analyzer to be applied.
    inputs: A list of `Column` or `Statistic`s to which the analyzer should
        be applied.
    args_dict: Extra arguments for the analyzer.
  """

  def __init__(self, tensor, analyzer_name, inputs, args_dict):
    super(_AnalyzerOutput, self).__init__(tensor)
    self._analyzer_name = analyzer_name
    self._inputs = inputs
    self._args_dict = args_dict

  @property
  def analyzer_name(self):
    return self._analyzer_name

  @property
  def inputs(self):
    return self._inputs

  @property
  def args_dict(self):
    return self._args_dict


class _InputColumn(Column):
  """A Column representing a column in the input dataset.

  Args:
    placeholder: The `Tensor` or `SparseTensor` that will represent the column.
    name: Name of the column (i.e. key in the dict of columns).
  """

  def __init__(self, placeholder, name):
    # In order to avoid a bug where import_graph_def fails when the input_map
    # and return_elements of an imported graph are the same (b/34288791), we
    # avoid using the placeholder of an input column as an output of a graph.
    # We do this by applying tf.identity to the placeholder and using the output
    # of tf.identity as the tensor representing the output of this column, thus
    # preventing the placeholder from being used as both an input and an output.
    if isinstance(placeholder, tf.SparseTensor):
      tensor = tf.SparseTensor(indices=tf.identity(placeholder.indices),
                               values=tf.identity(placeholder.values),
                               dense_shape=tf.identity(placeholder.dense_shape))
    else:
      tensor = tf.identity(placeholder)
    super(_InputColumn, self).__init__(tensor)
    self._name = name
    self._placeholder = placeholder

  @property
  def name(self):
    return self._name

  @property
  def placeholder(self):
    return self._placeholder


class _TransformedColumn(Column):
  """A Column containing the output of a transformation.

  A `_TransformedColumn` is defined by zero or more inputs (which may be
  `Column`s or `Statistic`s) and a function that accepts `Tensor`s or
  `SparseTensor`s as arguments and returns a `Tensor` or `SparseTensor`.

  Args:
    tensor: The `Tensor` or `SparseTensor` that will represent the column.
    fn: A function that accepts one or more `Tensor`s or `SparseTensor`s and
      returns a `Tensor` or `SparseTensor`.
    inputs: A list of `Column` or `Statistic`s to which the transform should
        be applied.
  """

  def __init__(self, tensor, fn, inputs):
    # Transforms are required to produce an output with a batch dimension. The
    # assertions below attempt to verify this. In the case of dense tensors the
    # check occurs statically if possible but falls back on a runtime check. In
    # the case of sparse tensors, the check happens at runtime.
    min_tensor_rank = 1
    if isinstance(tensor, tf.SparseTensor):
      with tf.control_dependencies(
          [tf.assert_greater_equal(tf.size(tensor.dense_shape),
                                   min_tensor_rank)]):
        tensor = tf.SparseTensor(indices=tf.identity(tensor.indices),
                                 values=tensor.values,
                                 dense_shape=tensor.dense_shape)
    else:
      with tf.control_dependencies(
          [tf.assert_rank_at_least(tensor, min_tensor_rank)]):
        tensor = tf.identity(tensor)
    super(_TransformedColumn, self).__init__(tensor)
    self._fn = fn
    self._inputs = inputs

  @property
  def fn(self):
    return self._fn

  @property
  def inputs(self):
    return self._inputs


# TODO(abrao): Consider adding helper class to avoid boilerplate for analyzer
# definitions.
def min(x):  # pylint: disable=redefined-builtin
  """Computes the minimum of a `Column`.

  Args:
    x: An input `Column'.

  Returns:
    A `Statistic` representing the minimum value of the input.
  """
  # TODO(kestert): A user needing to write code using the _AnalyzerOutput class
  # may not be the UX that we're looking for for writing custom analyzers.
  return _AnalyzerOutput(tf.placeholder(x.tensor.dtype, ()),
                         CanonicalAnalyzers.MIN, [x], {})


def max(x):  # pylint: disable=redefined-builtin
  """Computes the maximum of a `Column`.

  Args:
    x: An input `Column'.

  Returns:
    A `Statistic` representing the maximum value of the input.
  """
  return _AnalyzerOutput(tf.placeholder(x.tensor.dtype, ()),
                         CanonicalAnalyzers.MAX, [x], {})


def transform(fn, *args):
  """Applies a function to some columns.

  Applies a function to some columns given by the argument list. The number
  of arguments should match the number of inputs to the function. The args can
  also contain `Statistic`s in which case the values are broadcast across
  columns.

  Args:
    fn: A function that accepts one or more `Tensor`s or `SparseTensor`s and
      returns a `Tensor` or `SparseTensor`.
    *args: The list of `Column`s or `Statistic`s to apply the arguments to.

  Returns:
    A `Column` representing the application of the function.
  """
  input_tensors = [arg.tensor for arg in args]
  output_tensor = fn(*input_tensors)
  return _TransformedColumn(output_tensor, fn, args)


def scale_to_0_1(x):
  """Returns a column which is the input column scaled to have range [0,1].

  NOTE: this is just an example of how we might propagate metadata.

  Args:
    x: A `Column` representing a numeric value.

  Returns:
    A `Column` representing the input column scaled to [0, 1].
  """

  # A TITO function that scales x.
  def scale(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)

  out = transform(scale, x, min(x), max(x))
  out.add_metadata('min', 0)
  out.add_metadata('max', 1)

  return out


class CanonicalAnalyzers(object):
  """Enum-like class containing names of canonical analyzers."""

  MIN = 'min'
  MAX = 'max'


class AnalyzeDataset(object):
  """Takes a preprocessing_fn and computes the relevant statistics.

    AnalyzeDataset accepts a preprocessing_fn in its constructor.  When its
    `expand` method is called on a dataset, it computes all the relevant
    statistics required to run the transformation described by the
    preprocessing_fn, and returns a TransformFn representing the application of
    the preprocessing_fn.

    Args:
      preprocessing_fn: A function that accepts and returns a dictionary from
        strings to `Column`s or `Statistic`s.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, preprocessing_fn):
    self._preprocessing_fn = preprocessing_fn

  @property
  def preprocessing_fn(self):
    return self._preprocessing_fn

  @abc.abstractmethod
  def __ror__(self, dataset):
    """Syntactic sugar over `expand`.

    A typical implementation will implement this with self.expand(dataset),
    but may also do some additional implicit conversion of `dataset`.

    Args:
      dataset: The dataset on which one wants to run analysis.
    """
    pass

  @abc.abstractmethod
  def expand(self, dataset):
    """Analyze the dataset.

    Args:
      dataset: A dataset.

    Returns: A TransformFn computed from the input dataset.
    """
    pass


class TransformDataset(object):
  """Applies the transformation computed by transforming a Dataset.

  TransformDataset's `expand` method is called on a (dataset, transform_fn)
  pair. It applies the transform_fn to each row of the input dataset and
  returns the resulting dataset.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __ror__(self, dataset_and_transform_fn):
    """Syntactic sugar over expand.

    A typical implementation will implement this with
    self.expand(dataset_and_transform_fn), but may also do some additional
    implicit conversion of `dataset`.

    Args:
      dataset_and_transform_fn: A tuple of dataset and preprocessing
      function.
    """
    pass

  @abc.abstractmethod
  def expand(self, dataset_and_transform_fn):
    """Transforms the dataset using the transform_fn.

    Args:
      dataset_and_transform_fn: A tuple of dataset and preprocessing
      function.

    Returns:
      A dataset transformed according to the transform_fn.
    """
    pass


class AnalyzeAndTransformDataset(object):
  """Combination of AnalyzeDataset and TransformDataset.

  transformed, transform_fn = AnalyzeAndTransformDataset(
      preprocessing_fn).expand(dataset)

  should be equivalent to

  transform_fn = AnalyzeDataset(preprocessing_fn).expand(dataset)
  transformed = TransformDataset().expand((dataset, transform_fn))

  but may be more efficient since it avoids multiple passes over the data.

  Args:
    preprocessing_fn: A function that accepts and returns a dictionary from
        strings to `Column`s or `Statistic`s
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, preprocessing_fn):
    self._preprocessing_fn = preprocessing_fn

  @property
  def preprocessing_fn(self):
    return self._preprocessing_fn

  @abc.abstractmethod
  def __ror__(self, dataset):
    """Syntactic sugar over expand.

    A typical implementation will implement this with self.expand(dataset),
    but may also do some additional implicit conversion of `dataset`.

    Args:
      dataset: A dataset.
    """
    pass

  @abc.abstractmethod
  def expand(self, dataset):
    """Transform the dataset by applying the preprocessing_fn.

    Args:
      dataset: A dataset.

    Returns:
      A (Dataset, TransformFn) pair containing the preprocessed dataset and
      the graph that maps the input to the output data.
    """
    pass
