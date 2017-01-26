"""In-memory representation of the schema of a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Schema(object):
  """The schema of a dataset.

  This is an in-memory representation that may be serialized and deserialized to
  and from a variety of disk representations.
  """

  def __init__(self):
    self._features = {}

  def merge(self, other):
    # possible argument: resolution strategy (error or pick first and warn?)
    for key, value in other.features.items():
      if key in self.features:
        self.features[key].merge(value)
      else:
        self.features[key] = value

  # TODO(soergel): make this more immutable
  @property
  def features(self):
    # a dict of features
    return self._features
