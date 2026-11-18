# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace
import runpy


def test_mixed_rgbd_color_extensions(tmp_path):
    for folder in ('color', 'depth'):
        (tmp_path / folder).mkdir()
    for stem, suffix in [('00001', 'png'), ('00002', 'jpg'), ('00003', 'png')]:
        (tmp_path / 'color' / f'{stem}.{suffix}').touch()
        (tmp_path / 'depth' / f'{stem}.png').touch()

    examples = Path(__file__).resolve().parents[2] / 'examples' / 'python'
    legacy = runpy.run_path(str(examples / 'open3d_example.py'))
    tensor = runpy.run_path(str(examples / 't_reconstruction_system/common.py'))
    colors, depths = legacy['get_rgbd_file_lists'](str(tmp_path))
    config = SimpleNamespace(path_dataset=str(tmp_path),
                             depth_folder='depth',
                             color_folder='color')
    tensor_depths, tensor_colors = tensor['load_rgbd_file_names'](config)
    assert colors == tensor_colors
    assert depths == tensor_depths
    assert [Path(p).stem for p in colors] == [Path(p).stem for p in depths]
    assert len(colors) == 3

    # A complete set in a single format still takes priority over alternatives.
    (tmp_path / 'color' / '00002.png').touch()
    _, preferred_colors = tensor['load_rgbd_file_names'](config)
    assert [Path(p).suffix for p in preferred_colors] == ['.png'] * 3
