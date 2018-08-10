# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-memory representation of the schema of a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import six

import tensorflow as tf


_TF_EXAMPLE_ALLOWED_TYPES = [tf.string, tf.int64, tf.float32, tf.bool]


@contextlib.contextmanager
def _enter_column_context(name):
  try:
    yield
  except Exception as err:
    # Compatible with py3.
    err.args = (
        'Encountered an error while handling column "{}": '.format(name),) + (
            err.args or ('',))
    raise


class Schema(object):
  """The schema of a dataset.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.

  Args:
    column_schemas: A dict from logical column names to `ColumnSchema`s.
  """


  def __init__(self, column_schemas=None):
    if not column_schemas:
      column_schemas = {}
    if not isinstance(column_schemas, dict):
      raise ValueError('column_schemas must be a dict.')
    self._column_schemas = column_schemas

  @property
  def column_schemas(self):
    return self._column_schemas

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, repr(self.__dict__))

  def __getitem__(self, index):
    return self.column_schemas[index]


  def as_feature_spec(self):
    """Returns a representation of this Schema as a feature spec.

    A feature spec (for a whole dataset) is a dictionary from logical feature
    names to one of `FixedLenFeature`, `SparseFeature` or `VarLenFeature`.

    Returns:
      A representation of this Schema as a feature spec.
    """
    result = {}
    for key, column_schema in six.iteritems(self.column_schemas):
      with _enter_column_context(key):
        result[key] = column_schema.as_feature_spec()
    return result


class ColumnSchema(object):
  """The schema for a single column in a dataset.

  The schema contains two parts: the logical description of the column, which
  describes the nature of the actual data in the column (particularly this
  determines how this will ultimately be represented as a tensor) and the
  physical representation of the column, i.e. how the column's data is
  represented in memory or on disk.

  Fields:
    domain: a Domain object, providing the dtype and possibly other constraints.
    axes: a list of axes describing the intrinsic shape of the data,
      irrespective of its representation as dense or sparse.
    representation: A `ColumnRepresentation` that describes how the data is
        represented.
  """

  def __init__(self, domain, axes, representation):
    self.domain = domain
    self.axes = axes
    self._representation = representation

  @property
  def domain(self):
    return self._domain

  @domain.setter
  def domain(self, value):
    if not isinstance(value, Domain):
      value = _dtype_to_domain(value)
    self._domain = value

  @property
  def axes(self):
    return self._axes

  @axes.setter
  def axes(self, value):
    if not (value and isinstance(value[0], Axis)):
      value = _shape_to_axes(value)
    self._axes = value

  @property
  def representation(self):
    return self._representation

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, repr(self.__dict__))

  def as_feature_spec(self):
    """Returns a representation of this ColumnSchema as a feature spec.

    A feature spec (for a specific column) is one of a FixedLenFeature,
    SparseFeature or VarLenFeature.

    Returns:
      A representation of this ColumnSchema as a feature spec.
    """
    return self.representation.as_feature_spec(self)

  def tf_shape(self):
    """Represent the shape of this column as a `TensorShape`."""
    if self.axes is None:
      return tf.TensorShape(None)
    return tf.TensorShape([axis.size for axis in self.axes])

  def is_fixed_size(self):
    if self.axes is None:
      return False
    for axis in self.axes:
      if axis.size is None:
        return False
    return True


class Domain(object):
  """A description of the valid values that a column can take."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, dtype):
    self._dtype = tf.as_dtype(dtype)

  def __eq__(self, other):
    if other.__class__ == self.__class__:
      return self.__dict__ == other.__dict__
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, repr(self.__dict__))

  @property
  def dtype(self):
    return self._dtype

  # Serialize the tf.dtype as a string so that it can be unpickled on DataFlow.
  def __getstate__(self):
    return self._dtype.name

  def __setstate__(self, state):
    self._dtype = tf.as_dtype(state)




class FloatDomain(Domain):
  """A domain for a floating-point type."""

  def __init__(self, dtype):
    super(FloatDomain, self).__init__(dtype)
    if not self.dtype.is_floating:
      raise ValueError(
          'FloatDomain must be initialized with an floating point dtype.')


class IntDomain(Domain):
  """A domain for an integral type."""

  def __init__(self, dtype, min_value=None, max_value=None,
               is_categorical=None, vocabulary_file=''):
    super(IntDomain, self).__init__(dtype)
    if not self.dtype.is_integer:
      raise ValueError('IntDomain must be initialized with an integral dtype.')
    # NOTE: Because there is no uint64 or 128 bit ints, the following values
    # are always in the int64 range, which is important for the proto
    # representation.
    self._min_value = min_value if min_value is not None else self.dtype.min
    self._max_value = max_value if max_value is not None else self.dtype.max
    # Parsing a non-existing value from JSON will return None make sure it is
    # translated to False.
    self._is_categorical = (is_categorical
                            if is_categorical is not None
                            else False)
    self._vocabulary_file = vocabulary_file

  @property
  def min_value(self):
    return self._min_value

  @property
  def max_value(self):
    return self._max_value

  @property
  def is_categorical(self):
    return self._is_categorical

  @property
  def vocabulary_file(self):
    return self._vocabulary_file

  # Serialize the tf.dtype as a string so that it can be unpickled on DataFlow.
  def __getstate__(self):
    return {
        'dtype': self._dtype.name,
        'is_categorical': self._is_categorical,
        'min_value': self._min_value,
        'max_value': self._max_value,
        'vocabulary_file': self._vocabulary_file
    }

  def __setstate__(self, state):
    self._dtype = tf.as_dtype(state['dtype'])
    self._is_categorical = state['is_categorical']
    self._min_value = state['min_value']
    self._max_value = state['max_value']
    self._vocabulary_file = state['vocabulary_file']


class StringDomain(Domain):
  """A domain for a string type."""

  def __init__(self, dtype):
    super(StringDomain, self).__init__(dtype)
    if self.dtype != tf.string:
      raise ValueError('StringDomain must be initialized with a string dtype.')


class BoolDomain(Domain):
  """A domain for a boolean type."""

  def __init__(self, dtype):
    super(BoolDomain, self).__init__(dtype)
    if self.dtype != tf.bool:
      raise ValueError('BoolDomain must be initialized with a boolean dtype.')


def _dtype_to_domain(dtype):
  """Create an appropriate Domain for the given dtype."""
  if dtype.is_integer:
    return IntDomain(dtype)
  if dtype.is_floating:
    return FloatDomain(dtype)
  if dtype == tf.string:
    return StringDomain(dtype)
  if dtype == tf.bool:
    return BoolDomain(dtype)
  raise ValueError('Schema cannot accommodate dtype: {}'.format(dtype))


class Axis(object):
  """An axis representing one dimension of the shape of a column.

  Elements are:
    size: integer.  The length of the axis.  None = unknown.
  """

  def __init__(self, size):
    self._size = size

  @property
  def size(self):
    return self._size

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, repr(self.__dict__))


class ColumnRepresentation(object):
  """A description of the representation of a column in memory or on disk."""

  __metaclass__ = abc.ABCMeta

  def __eq__(self, other):
    if other.__class__ == self.__class__:
      return self.__dict__ == other.__dict__
    return NotImplemented

  def __ne__(self, other):
    return not self == other

  @abc.abstractmethod
  def as_feature_spec(self, column):
    """Returns the representation of this column as a feature spec.

    Args:
      column: The column to be represented.
    """
    raise NotImplementedError()

# note we don't provide tf.FixedLenSequenceFeature yet, because that is
# only used to parse tf.SequenceExample.


class FixedColumnRepresentation(ColumnRepresentation):
  """Represent the column using a fixed size."""

  def __init__(self, default_value=None):
    super(FixedColumnRepresentation, self).__init__()
    self._default_value = default_value

  @property
  def default_value(self):
    """Default value may be None, but then missing data produces an error."""
    return self._default_value

  def __repr__(self):
    return '%s(%r)' % (self.__class__.__name__, self._default_value)

  def as_feature_spec(self, column):
    if not column.is_fixed_size():
      raise ValueError('A column of unknown size cannot be represented as '
                       'fixed-size.')
    if column.domain.dtype not in _TF_EXAMPLE_ALLOWED_TYPES:
      raise ValueError('tf.Example parser supports only types {}, so it is '
                       'invalid to generate a feature_spec with type '
                       '{}.'.format(
                           _TF_EXAMPLE_ALLOWED_TYPES,
                           repr(column.domain.dtype)))
    return tf.FixedLenFeature(column.tf_shape().as_list(),
                              column.domain.dtype,
                              self.default_value)


class ListColumnRepresentation(ColumnRepresentation):
  """Represent the column using a variable size."""

  def __init__(self):
    super(ListColumnRepresentation, self).__init__()

  def __repr__(self):
    return '%s()' % (self.__class__.__name__,)

  def as_feature_spec(self, column):
    if column.domain.dtype not in _TF_EXAMPLE_ALLOWED_TYPES:
      raise ValueError('tf.Example parser supports only types {}, so it is '
                       'invalid to generate a feature_spec with type '
                       '{}.'.format(
                           _TF_EXAMPLE_ALLOWED_TYPES,
                           repr(column.domain.dtype)))
    return tf.VarLenFeature(column.domain.dtype)


class SparseColumnRepresentation(ColumnRepresentation):
  """Sparse physical representation of a logically fixed-size column."""

  def __init__(self, value_field_name, index_fields):
    super(SparseColumnRepresentation, self).__init__()
    self._value_field_name = value_field_name
    self._index_fields = index_fields

  @property
  def value_field_name(self):
    return self._value_field_name

  @property
  def index_fields(self):
    # SparseIndexes
    return self._index_fields

  def __repr__(self):
    return '%s(%r, %r)' % (self.__class__.__name__,
                           self._value_field_name, self._index_fields)

  def as_feature_spec(self, column):
    ind = self.index_fields
    if len(ind) != 1 or len(column.axes) != 1:
      raise ValueError('tf.Example parser supports only 1-d sparse features.')
    index = ind[0]

    if column.domain.dtype not in _TF_EXAMPLE_ALLOWED_TYPES:
      raise ValueError('tf.Example parser supports only types {}, so it is '
                       'invalid to generate a feature_spec with type '
                       '{}.'.format(
                           _TF_EXAMPLE_ALLOWED_TYPES,
                           repr(column.domain.dtype)))

    return tf.SparseFeature(index.name,
                            self._value_field_name,
                            column.domain.dtype,
                            column.axes[0].size,
                            index.is_sorted)


class SparseIndexField(collections.namedtuple('SparseIndexField',
                                              ['name', 'is_sorted'])):
  pass


def from_feature_spec(feature_spec):
  """Convert a feature_spec to a Schema.

  Args:
    feature_spec: a features specification in the format expected by
        tf.parse_example(), i.e.
        `{name: FixedLenFeature(...), name: VarLenFeature(...), ...'

  Returns:
    A Schema representing the provided set of columns.
  """
  return Schema({
      key: _from_parse_feature(parse_feature)
      for key, parse_feature in six.iteritems(feature_spec)
  })


def _from_parse_feature(parse_feature):
  """Convert a single feature spec to a ColumnSchema."""

  # FixedLenFeature
  if isinstance(parse_feature, tf.FixedLenFeature):
    representation = FixedColumnRepresentation(parse_feature.default_value)
    return ColumnSchema(parse_feature.dtype, parse_feature.shape,
                        representation)

  # FixedLenSequenceFeature
  if isinstance(parse_feature, tf.FixedLenSequenceFeature):
    raise ValueError('DatasetSchema does not support '
                     'FixedLenSequenceFeature yet.')

  # VarLenFeature
  if isinstance(parse_feature, tf.VarLenFeature):
    representation = ListColumnRepresentation()
    return ColumnSchema(parse_feature.dtype, [None], representation)

  # SparseFeature
  if isinstance(parse_feature, tf.SparseFeature):
    index_field = SparseIndexField(name=parse_feature.index_key,
                                   is_sorted=parse_feature.already_sorted)
    representation = SparseColumnRepresentation(
        value_field_name=parse_feature.value_key,
        index_fields=[index_field])
    return ColumnSchema(parse_feature.dtype, [parse_feature.size],
                        representation)

  raise ValueError('Cannot interpret feature spec {} with type {}'.format(
      parse_feature, type(parse_feature)))


def infer_column_schema_from_tensor(tensor):
  """Infer a ColumnSchema from a tensor."""
  if isinstance(tensor, tf.SparseTensor):
    # For SparseTensor, there's insufficient information to distinguish between
    # ListColumnRepresentation and SparseColumnRepresentation. So we just guess
    # the former, and callers are expected to handle the latter case on their
    # own (e.g. by requiring the user to provide the schema). This is a policy
    # motivated by the prevalence of VarLenFeature in current tf.Learn code.
    axes = [Axis(None)]
    representation = ListColumnRepresentation()
  else:
    axes = _shape_to_axes(tensor.get_shape(),
                          remove_batch_dimension=True)
    representation = FixedColumnRepresentation()
  return ColumnSchema(tensor.dtype, axes, representation)


def _shape_to_axes(shape, remove_batch_dimension=False):
  """Create axes for the given shape.

  Args:
    shape: A list of axis sizes, or a `TensorShape`.
    remove_batch_dimension: A boolean indicating whether to remove the 0th
      dimension.

  Returns:
    A list of axes representing the given shape.

  Raises:
    ValueError: If `remove_batch_dimension` is True and the given shape does not
      have rank >= 1.
  """
  if shape is None or (isinstance(shape, tf.TensorShape)
                       and shape.dims is None):
    axes = None
  else:
    if isinstance(shape, tf.TensorShape):
      shape = shape.as_list()
    axes = [Axis(axis_size) for axis_size in shape]
    if remove_batch_dimension:
      if len(axes) < 1:
        raise ValueError('Expected tf_shape to have rank >= 1')
      axes = axes[1:]
  return axes
