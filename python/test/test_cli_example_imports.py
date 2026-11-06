# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import argparse
import sys

from open3d.tools import cli


def test_example_can_import_sibling(tmp_path, monkeypatch):
    name = 'o3d_cli_local_example_dep'
    category = tmp_path / 'geometry'
    category.mkdir()
    (category / (name + '.py')).write_text('value = 42')
    output = tmp_path / 'result.txt'
    (category / 'example.py').write_text(
        f'from {name} import value\n'
        'from pathlib import Path\n'
        'import sys\n'
        'Path(sys.argv[1]).write_text(str(value))')
    monkeypatch.setattr(cli, '_get_examples_dir', lambda: tmp_path)
    monkeypatch.setattr(cli, '_get_runnable_examples_dict',
                        lambda: {'geometry': ['example']})
    argv = ['open3d', 'example', 'geometry/example', str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    before = sys.path.copy()
    args = argparse.Namespace(category_example='geometry/example',
                              list=False,
                              show=False)
    try:
        assert cli._example(argparse.ArgumentParser(), args) == 0
        assert output.read_text() == '42'
        assert sys.path == before
        assert sys.argv == argv
    finally:
        sys.modules.pop(name, None)
