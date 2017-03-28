# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Package Setup script for the tf.Transform binary.
"""
import os

from setuptools import find_packages
from setuptools import setup


def get_required_install_packages():
  return [

      # Using >= for better integration tests. During release this is
      # automatically changed to a ==.
      'google-cloud-dataflow == 0.6.0',
  ]


def get_version():
  # Obtain the version from the global names on version.py
  # We cannot do 'from tensorflow_transform import version' since the transitive
  # dependencies will not be available when the installer is created.
  global_names = {}
  execfile(os.path.normpath('tensorflow_transform/version.py'), global_names)
  return global_names['__version__']


setup(
    name='tensorflow-transform',
    version=get_version(),
    author='Google Inc.',
    author_email='tf-transform-feedback@google.com',
    license='Apache 2.0',
    namespace_packages=[],
    install_requires=get_required_install_packages(),
    packages=find_packages(),
    include_package_data=True,
    description='A library for data preprocessing with TensorFlow',
    requires=[])
