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
"""Package Setup script for tf.Transform."""
import os

from setuptools import find_packages
from setuptools import setup


def select_constraint(default, nightly=None, git_master=None):
  """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
  selector = os.environ.get('TFX_DEPENDENCY_SELECTOR')
  if selector == 'UNCONSTRAINED':
    return ''
  elif selector == 'NIGHTLY' and nightly is not None:
    return nightly
  elif selector == 'GIT_MASTER' and git_master is not None:
    return git_master
  else:
    return default


# Get version from version module.
with open('tensorflow_transform/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF and pyarrow version with tfx-bsl.
  return [
      'absl-py>=0.9,<0.13',
      'apache-beam[gcp]>=2.29,<3',
      'numpy>=1.16,<1.20',
      'protobuf>=3.9.2,<4',
      'pyarrow>=1,<3',
      'pydot>=1.2,<2',
      'six>=1.12,<2',
      'tensorflow' + select_constraint(
          '>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<2.6'),
      'tensorflow-metadata' + select_constraint(
          default='>=0.30,<0.31',
          nightly='>=0.31.0.dev',
          git_master='@git+https://github.com/tensorflow/metadata@master'),
      'tfx-bsl' + select_constraint(
          default='>=0.30,<0.31',
          nightly='>=0.31.0.dev',
          git_master='@git+https://github.com/tensorflow/tfx-bsl@master'),
  ]


# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tensorflow-transform',
    version=__version__,
    author='Google Inc.',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    python_requires='>=3.6,<4',
    packages=find_packages(),
    include_package_data=True,
    description='A library for data preprocessing with TensorFlow',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow transform tfx',
    url='https://www.tensorflow.org/tfx/transform/get_started',
    download_url='https://github.com/tensorflow/transform/tags',
    requires=[])
