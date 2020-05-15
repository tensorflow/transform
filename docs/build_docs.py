# Lint as: python3
#
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
r"""Generate docs for tft.

This requires a local installation of `tft` and `tensoirflow_docs`

```
$ pip install tensorflow_transform git+https://github.com/tensorflow/docs
```

```
python build_docs.py --output_dir=/tmp/tft-api
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib
import shutil
import tempfile


from absl import app
from absl import flags

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api
import tensorflow_transform as transform
import tensorflow_transform.beam as tft_beam

import yaml

flags.DEFINE_string('output_dir', '/tmp/tft_api/',
                    'The path to output the files to')

flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/transform/tree/master/tensorflow_transform',
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'tfx/transform/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS


def main(args):
  print()
  if args[1:]:
    raise ValueError('Unrecognized Command line args', args[1:])

  tft_out = pathlib.Path(tempfile.mkdtemp())

  doc_controls.do_not_generate_docs(tft_beam.analyzer_impls)

  doc_generator = generate_lib.DocGenerator(
      root_title='TF-Transform',
      py_modules=[('tft', transform)],
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      callbacks=[public_api.explicit_package_contents_filter])

  doc_generator.build(tft_out)

  tft_beam_out = pathlib.Path(tempfile.mkdtemp())
  doc_generator = generate_lib.DocGenerator(
      root_title='TFT-Beam',
      py_modules=[('tft_beam', tft_beam)],
      code_url_prefix=FLAGS.code_url_prefix + '/beam',
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      callbacks=[
          public_api.explicit_package_contents_filter,
          public_api.local_definitions_filter
      ])

  doc_generator.build(tft_beam_out)

  output_dir = pathlib.Path(FLAGS.output_dir)

  def splice(name, tmp_dir):
    shutil.rmtree(output_dir / name, ignore_errors=True)
    shutil.copytree(tmp_dir / name, output_dir / name)
    shutil.copy(tmp_dir / f'{name}.md', output_dir / f'{name}.md')
    try:
      shutil.copy(tmp_dir / '_redirects.yaml',
                  output_dir / name / '_redirects.yaml')
    except FileNotFoundError:
      pass
    shutil.copy(tmp_dir / '_toc.yaml', output_dir / name / '_toc.yaml')

  splice('tft', tft_out)
  splice('tft_beam', tft_beam_out)

  toc_path = output_dir / '_toc.yaml'
  toc_text = yaml.dump(
      {'toc': [
          {'include': f'{FLAGS.site_path}/tft/_toc.yaml'},
          {'break': True},
          {'include': f'{FLAGS.site_path}/tft/_toc.yaml'}]})
  toc_path.write_text(toc_text)


if __name__ == '__main__':
  app.run(main)
