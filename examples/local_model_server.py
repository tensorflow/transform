# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Local model server for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib


def local_model_server_supported():
  return False


@contextlib.contextmanager
def start_server(model_name, model_path):
  del model_name  # unused
  del model_path  # unused
  raise NotImplementedError


# TODO(KesterTong): Change the input of make_classification_request to not be a
# string. This will require adding a test-only dependency on
# tensorflow_serving.apis.
def make_classification_request(address, ascii_classification_request):
  """Makes a classify request to a local server."""
  del address  # unused
  del ascii_classification_request  # unused
  raise NotImplementedError
