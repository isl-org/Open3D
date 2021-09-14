# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
import os
import tempfile
from time import sleep
import subprocess as sp
import webbrowser
import shutil
import numpy as np
import pytest
pytest.importorskip("tensorboard")

import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from open3d.visualization.tensorboard_plugin.util import Open3DPluginDataReader

from open3d_test import test_data_dir


@pytest.fixture
def geometry_data():
    """Common geometry data for tests"""
    cube = (o3d.geometry.TriangleMesh.create_box(1, 2, 4),
            o3d.geometry.TriangleMesh.create_box(1, 2, 4))
    cube[0].compute_vertex_normals()
    cube[1].compute_vertex_normals()

    cube_ls = tuple(
        o3d.geometry.LineSet.create_from_triangle_mesh(c) for c in cube)

    colors = (((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
              ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)), ((0.0, 0.0, 1.0), (1.0, 1.0,
                                                                     0.0)))

    return {
        'cube': cube,
        'cube_ls': cube_ls,
        'colors': colors,
        'max_outputs': 2
    }


def test_tensorflow_summary(geometry_data):
    """Test writing summary from TensorFlow
    """

    tf = pytest.importorskip("tensorflow")
    logdir = tempfile.mkdtemp(prefix='open3d_tb_plugin_test')
    writer = tf.summary.create_file_writer(logdir)

    rng = np.random
    tensor_converter = (tf.convert_to_tensor, o3d.core.Tensor.from_numpy,
                        np.array)

    cube = geometry_data['cube']
    cube_ls = geometry_data['cube_ls']
    colors = geometry_data['colors']
    max_outputs = geometry_data['max_outputs']
    with writer.as_default():
        for step in range(3):
            cube[0].paint_uniform_color(colors[step][0])
            cube[1].paint_uniform_color(colors[step][1])
            # Randomly convert to TF, Open3D, Numpy tensors, or use property
            # reference
            cube_summary = to_dict_batch(cube)
            if step > 0:
                cube_summary['vertex_positions'] = 0
                cube_summary['vertex_normals'] = 0
                cube_summary['vertex_colors'] = rng.choice(tensor_converter)(
                    cube_summary['vertex_colors'])
            else:
                for prop, tensor in cube_summary.items():
                    cube_summary[prop] = rng.choice(tensor_converter)(tensor)
            summary.add_3d('cube',
                           cube_summary,
                           step=step,
                           logdir=logdir,
                           max_outputs=max_outputs)
            cube_summary.pop('triangle_indices')  # Convert to PointCloud
            summary.add_3d('cube_pcd',
                           cube_summary,
                           step=step,
                           logdir=logdir,
                           max_outputs=max_outputs)
            cube_ls[0].paint_uniform_color(colors[step][0])
            cube_ls[1].paint_uniform_color(colors[step][1])
            cube_ls_summary = to_dict_batch(cube_ls)
            for prop, tensor in cube_ls_summary.items():
                cube_ls_summary[prop] = rng.choice(tensor_converter)(tensor)
            summary.add_3d('cube_ls',
                           cube_ls_summary,
                           step=step,
                           logdir=logdir,
                           max_outputs=max_outputs)

    sleep(0.25)  # msgpack writing disk flush time
    dirpath_ref = [
        logdir,
        os.path.join(logdir, 'plugins'),
        os.path.join(logdir, 'plugins/Open3D')
    ]
    filenames_ref = [['events.out.tfevents.*'], [], ['cube.*.msgpack']]

    dirpath, filenames = [], []
    for dp, unused_dn, fn in os.walk(logdir):
        dirpath.append(dp)
        filenames.append(fn)

    assert (dirpath[:2] == dirpath_ref[:2] and
            dirpath[2][0][:20] == dirpath_ref[2][0][:20])
    assert filenames[0][0][:20] == filenames_ref[0][0][:20]
    assert set(x.split('.')[0] for x in filenames[2]) == set(
        ('cube', 'cube_pcd', 'cube_ls'))
    assert filenames_ref[2][0][-8:] == '.msgpack'

    # Note: The event file written during this test cannot be reliably verified
    # in the same Python process, since it's usually buffered by GFile / Python
    # / OS and written to disk in increments of the filesystem blocksize.
    # Complete write is guaranteed after Python has exited.
    shutil.rmtree(logdir)


def test_pytorch_summary(geometry_data):
    """Test writing summary from PyTorch"""

    torch = pytest.importorskip("torch")
    torch_tb = pytest.importorskip("torch.utils.tensorboard")
    SummaryWriter = torch_tb.SummaryWriter
    logdir = tempfile.mkdtemp(prefix='open3d_tb_plugin_test')
    writer = SummaryWriter(logdir)

    rng = np.random
    tensor_converter = (torch.from_numpy, o3d.core.Tensor.from_numpy, np.array)

    cube = geometry_data['cube']
    cube_ls = geometry_data['cube_ls']
    colors = geometry_data['colors']
    max_outputs = geometry_data['max_outputs']
    for step in range(3):
        cube[0].paint_uniform_color(colors[step][0])
        cube[1].paint_uniform_color(colors[step][1])
        cube_summary = to_dict_batch(cube)
        # Randomly convert to PyTorch, Open3D, Numpy tensors, or use property
        # reference
        if step > 0:
            cube_summary['vertex_positions'] = 0
            cube_summary['vertex_normals'] = 0
            cube_summary['vertex_colors'] = rng.choice(tensor_converter)(
                cube_summary['vertex_colors'])
        else:
            for prop, tensor in cube_summary.items():
                cube_summary[prop] = rng.choice(tensor_converter)(tensor)
        writer.add_3d('cube', cube_summary, step=step, max_outputs=max_outputs)
        cube_summary.pop('triangle_indices')  # Convert to PointCloud
        writer.add_3d('cube_pcd',
                      cube_summary,
                      step=step,
                      max_outputs=max_outputs)
        cube_ls[0].paint_uniform_color(colors[step][0])
        cube_ls[1].paint_uniform_color(colors[step][1])
        cube_ls_summary = to_dict_batch(cube_ls)
        for prop, tensor in cube_ls_summary.items():
            cube_ls_summary[prop] = rng.choice(tensor_converter)(tensor)
        writer.add_3d('cube_ls',
                      cube_ls_summary,
                      step=step,
                      max_outputs=max_outputs)

    sleep(0.25)  # msgpack writing disk flush time
    dirpath_ref = [
        logdir,
        os.path.join(logdir, 'plugins'),
        os.path.join(logdir, 'plugins/Open3D')
    ]
    filenames_ref = [['events.out.tfevents.*'], [], ['cube.*.msgpack']]

    dirpath, filenames = [], []
    for dp, unused_dn, fn in os.walk(logdir):
        dirpath.append(dp)
        filenames.append(fn)

    assert (dirpath[:2] == dirpath_ref[:2] and
            dirpath[2][0][:20] == dirpath_ref[2][0][:20])
    assert filenames[0][0][:20] == filenames_ref[0][0][:20]
    assert set(x.split('.')[0] for x in filenames[2]) == set(
        ('cube', 'cube_pcd', 'cube_ls'))
    assert filenames_ref[2][0][-8:] == '.msgpack'

    # Note: The event file written during this test cannot be reliably verified
    # in the same Python process, since it's usually buffered by GFile / Python
    # / OS and written to disk in increments of the filesystem blocksize.
    # Complete write is guaranteed after Python has exited.
    shutil.rmtree(logdir)


def test_plugin_data_reader(geometry_data):
    """Test reading summary data"""
    shutil.unpack_archive(
        os.path.join(test_data_dir, "test_tensorboard_plugin.zip"))
    logdir = "test_tensorboard_plugin"
    cube = geometry_data['cube']
    cube_ls = geometry_data['cube_ls']
    colors = geometry_data['colors']
    max_outputs = geometry_data['max_outputs']

    reader = Open3DPluginDataReader(logdir)
    assert reader.is_active()
    assert reader.run_to_tags == {'.': ['cube', 'cube_pcd', 'cube_ls']}
    step_to_idx = {i: i for i in range(3)}
    for step in range(3):
        for batch_idx in range(max_outputs):
            cube[batch_idx].paint_uniform_color(colors[step][batch_idx])
            cube_ref = o3d.t.geometry.TriangleMesh.from_legacy(cube[batch_idx])
            cube_ref.triangle["indices"] = cube_ref.triangle["indices"].to(
                o3d.core.int32)
            cube_ref.vertex['colors'] = (cube_ref.vertex['colors'] * 255).to(
                o3d.core.uint8)

            cube_out = reader.read_geometry(".", "cube", step, batch_idx,
                                            step_to_idx)
            assert (cube_out.vertex['positions'] == cube_ref.vertex['positions']
                   ).all()
            assert (
                cube_out.vertex['normals'] == cube_ref.vertex['normals']).all()
            assert (
                cube_out.vertex['colors'] == cube_ref.vertex['colors']).all()
            assert (cube_out.triangle['indices'] == cube_ref.triangle['indices']
                   ).all()

            cube_pcd_out = reader.read_geometry(".", "cube_pcd", step,
                                                batch_idx, step_to_idx)
            assert (cube_pcd_out.point['positions'] ==
                    cube_ref.vertex['positions']).all()
            assert (cube_pcd_out.point['normals'] == cube_ref.vertex['normals']
                   ).all()
            assert (cube_pcd_out.point['colors'] == cube_ref.vertex['colors']
                   ).all()

            cube_ls[batch_idx].paint_uniform_color(colors[step][batch_idx])
            cube_ls_ref = o3d.t.geometry.LineSet.from_legacy(cube_ls[batch_idx])
            cube_ls_ref.line["indices"] = cube_ls_ref.line["indices"].to(
                o3d.core.int32)
            cube_ls_ref.line['colors'] = (cube_ls_ref.line['colors'] * 255).to(
                o3d.core.uint8)

            cube_ls_out = reader.read_geometry(".", "cube_ls", step, batch_idx,
                                               step_to_idx)
            assert (cube_ls_out.point['positions'] ==
                    cube_ls_ref.point['positions']).all()
            assert (cube_ls_out.line['indices'] == cube_ls_ref.line['indices']
                   ).all()
            assert (
                cube_ls_out.line['colors'] == cube_ls_ref.line['colors']).all()

    shutil.rmtree(logdir)


@pytest.mark.skip(reason="This will only run on a local machine with GPU.")
def test_tensorboard_app():
    shutil.unpack_archive(
        os.path.join(test_data_dir, "test_tensorboard_plugin.zip"))
    logdir = "test_tensorboard_plugin"
    with sp.Popen(['tensorboard', '--logdir', logdir]) as tb_proc:
        sleep(5)
        webbrowser.open('http://localhost:6006/')
        sleep(5)
        tb_proc.terminate()
        sleep(2)
        tb_proc.kill()
    shutil.rmtree(logdir)
