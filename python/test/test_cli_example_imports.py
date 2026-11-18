# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import argparse
import sys

import pytest

from open3d.tools import cli


@pytest.fixture
def example_dir(tmp_path, monkeypatch):
    category = tmp_path / 'geometry'
    category.mkdir()
    monkeypatch.setattr(cli, '_get_examples_dir', lambda: tmp_path)
    monkeypatch.setattr(cli, '_get_runnable_examples_dict',
                        lambda: {'geometry': ['example']})
    return category


def test_example_can_import_sibling(tmp_path, monkeypatch, example_dir):
    name = 'o3d_cli_local_example_dep'
    (example_dir / (name + '.py')).write_text('value = 42')
    output = tmp_path / 'result.txt'
    (example_dir / 'example.py').write_text(
        f'from {name} import value\n'
        'from pathlib import Path\n'
        'import sys\n'
        'Path(sys.argv[1]).write_text(str(value))')
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


@pytest.mark.parametrize('show', [False, True], ids=['raises', 'show'])
def test_example_preserves_caller_state(example_dir, monkeypatch, capsys, show):
    source = "raise RuntimeError('example failed')"
    (example_dir / 'example.py').write_text(source)
    argv = ['open3d', 'example', 'geometry/example']
    path = sys.path.copy()
    monkeypatch.setattr(sys, 'argv', argv.copy())
    monkeypatch.setattr(sys, 'path', path.copy())
    caller_argv, caller_path = sys.argv, sys.path
    args = argparse.Namespace(category_example='geometry/example',
                              list=False,
                              show=show)
    if show:
        assert cli._example(argparse.ArgumentParser(), args) == 0
        assert capsys.readouterr().out.strip() == source
    else:
        with pytest.raises(RuntimeError, match='example failed'):
            cli._example(argparse.ArgumentParser(), args)
    assert sys.argv is caller_argv
    assert sys.argv == argv
    assert sys.path is caller_path
    assert sys.path == path
