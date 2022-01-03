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
import sys
from time import sleep
import subprocess as sp
import webbrowser
import shutil
import numpy as np
import pytest
pytest.importorskip("tensorboard")
vis = pytest.importorskip("open3d.ml.vis")
try:
    BoundingBox3D = vis.BoundingBox3D
except AttributeError:
    pytestmark = pytest.mark.skip(reason="BoundingBox3D not available.")

import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from open3d.visualization.tensorboard_plugin.util import Open3DPluginDataReader

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import test_data_dir


@pytest.fixture
def geometry_data():
    """Common geometry data for tests"""
    cube = (o3d.geometry.TriangleMesh.create_box(1, 2, 4, create_uv_map=True),
            o3d.geometry.TriangleMesh.create_box(1, 2, 4, create_uv_map=True))
    cube[0].compute_vertex_normals()
    cube[1].compute_vertex_normals()

    n_vertices = 8
    n_dims = 4
    cube_custom_prop = tuple(
        np.linspace(
            0, step, num=len(cube) * n_vertices *
            n_dims, dtype=np.float32).reshape((len(cube), n_vertices, n_dims))
        for step in range(3))
    label_to_names = {
        -1: 'unknown',  # negative label
        0: 'ground',
        1: 'sky',
        3: 'water',  # non-consecutive
        5: 'fire',
        7: 'space'
    }
    labels = tuple(label_to_names.keys())
    cube_labels = tuple(
        tuple(
            np.full((n_vertices, 1), labels[step * 2 + batch_idx], dtype=int)
            for batch_idx in range(2))
        for step in range(3))

    cube_ls = tuple(
        o3d.geometry.LineSet.create_from_triangle_mesh(c) for c in cube)

    colors = (((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
              ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)), ((0.0, 0.0, 1.0), (1.0, 1.0,
                                                                     0.0)))
    material = {
        "material_name": ("defaultLit", "defaultUnlit"),
        "material_scalar_point_size": (2, 20),
        "material_scalar_metallic": (0.25, 0.75),
        "material_vector_base_color": (
            (0.25, 0.25, 0.25, 1.0), (0.25, 0.25, 0.25, 1.0)),
        "material_texture_map_metallic":
            np.full((2, 8, 8, 1), 128, dtype=np.uint8),
        "material_texture_map_albedo":  # albedo = 64 is fairly dark
            np.full((2, 8, 8, 3), 64, dtype=np.uint8),
    }

    material_ls = {
        "material_name": ("unlitLine", "unlitLine"),
        "material_scalar_line_width": (2, 20),
        "material_vector_base_color": (
            (0.25, 0.25, 0.25, 1.0), (0.25, 0.25, 0.25, 1.0))
    }

    bboxes = []
    for step in range(3):
        bboxes.append([])
        for batch_idx in range(2):
            nbb = step * 2 + batch_idx + 1
            center = np.linspace(-nbb, nbb, num=3 * nbb).reshape((nbb, 3))
            size = np.linspace(nbb, 4 * nbb, num=3 * nbb).reshape((nbb, 3))
            label_class = list(labels[k] for k in range(nbb))
            confidence = np.linspace(0., 1., num=nbb)
            bboxes[-1].append(
                tuple(
                    BoundingBox3D(center[k], (0, 0, 1), (0, 1, 0), (
                        1, 0, 0), size[k], label_class[k], confidence[k])
                    for k in range(nbb)))

    tags = ['cube', 'cube_pcd', 'cube_ls']
    filenames = [['events.out.tfevents.*'], [], ['cube.*.msgpack'],
                 ['cube_ls.*.msgpack'], ['cube_pcd.*.msgpack']]
    if len(bboxes) > 0:
        tags.append('bboxes')
        filenames.append(['bboxes.*.msgpack'])
    return {
        'cube': cube,
        'material': material,
        'cube_ls': cube_ls,
        'material_ls': material_ls,
        'colors': colors,
        'cube_custom_prop': cube_custom_prop,
        'cube_labels': cube_labels,
        'label_to_names': label_to_names,
        'bboxes': bboxes,
        'max_outputs': 2,
        'tags': sorted(tags),
        'filenames': filenames
    }


def test_tensorflow_summary(geometry_data, tmp_path):
    """Test writing summary from TensorFlow"""
    tf = pytest.importorskip("tensorflow")
    logdir = str(tmp_path)
    writer = tf.summary.create_file_writer(logdir)

    rng = np.random.default_rng()
    tensor_converter = (tf.convert_to_tensor, o3d.core.Tensor.from_numpy,
                        np.array)

    cube, material = geometry_data['cube'], geometry_data['material']
    cube_custom_prop = geometry_data['cube_custom_prop']
    cube_ls, material_ls = geometry_data['cube_ls'], geometry_data[
        'material_ls']
    colors = geometry_data['colors']
    cube_labels = geometry_data['cube_labels']
    label_to_names = geometry_data['label_to_names']
    max_outputs = geometry_data['max_outputs']
    bboxes = geometry_data['bboxes']
    with writer.as_default():
        for step in range(3):
            cube[0].paint_uniform_color(colors[step][0])
            cube[1].paint_uniform_color(colors[step][1])
            cube_summary = to_dict_batch(cube)
            cube_summary.update(material)
            # Randomly convert to TF, Open3D, Numpy tensors, or use property
            # reference
            if step > 0:
                cube_summary['vertex_positions'] = 0  # step ref.
                cube_summary['vertex_normals'] = 0
                cube_summary['vertex_colors'] = rng.choice(tensor_converter)(
                    cube_summary['vertex_colors'])
                label_to_names = None  # Only need for first step
            else:
                for prop, tensor in cube_summary.items():
                    # skip material scalar and vector props
                    if (not prop.startswith("material_") or
                            prop.startswith("material_texture_map_")):
                        cube_summary[prop] = rng.choice(tensor_converter)(
                            tensor)
            summary.add_3d('cube',
                           cube_summary,
                           step=step,
                           logdir=logdir,
                           max_outputs=max_outputs)
            for key in tuple(cube_summary):  # Convert to PointCloud
                if key.startswith(('triangle_', 'material_texture_map_')):
                    cube_summary.pop(key)
            cube_summary['vertex_custom'] = tuple(
                rng.choice(tensor_converter)(tensor)
                for tensor in cube_custom_prop[step])  # Add custom prop
            cube_summary['vertex_labels'] = tuple(
                rng.choice(tensor_converter)(tensor)
                for tensor in cube_labels[step])  # Add labels
            summary.add_3d('cube_pcd',
                           cube_summary,
                           step=step,
                           logdir=logdir,
                           max_outputs=max_outputs,
                           label_to_names=label_to_names)
            cube_ls[0].paint_uniform_color(colors[step][0])
            cube_ls[1].paint_uniform_color(colors[step][1])
            cube_ls_summary = to_dict_batch(cube_ls)
            cube_ls_summary.update(material_ls)
            for prop, tensor in cube_ls_summary.items():
                if (not prop.startswith("material_") or
                        prop.startswith("material_texture_map_")):
                    cube_ls_summary[prop] = rng.choice(tensor_converter)(tensor)
            summary.add_3d('cube_ls',
                           cube_ls_summary,
                           step=step,
                           logdir=logdir,
                           max_outputs=max_outputs)
            if len(bboxes) > 0:
                summary.add_3d('bboxes', {'bboxes': bboxes[step]},
                               step=step,
                               logdir=logdir,
                               max_outputs=max_outputs,
                               label_to_names=label_to_names)

    sleep(0.25)  # msgpack writing disk flush time
    tags_ref = geometry_data['tags']
    dirpath_ref = [
        logdir,
        os.path.join(logdir, 'plugins'),
        os.path.join(logdir, 'plugins/Open3D')
    ]
    filenames_ref = geometry_data['filenames']

    dirpath, filenames = [], []
    for dp, unused_dn, fn in os.walk(logdir):
        dirpath.append(dp)
        filenames.append(fn)

    assert dirpath == dirpath_ref
    assert filenames[0][0].startswith(filenames_ref[0][0][:20])
    assert sorted(x.split('.')[0] for x in filenames[2]) == tags_ref
    assert all(fn.endswith('.msgpack') for fn in filenames[2])
    # Note: The event file written during this test cannot be reliably verified
    # in the same Python process, since it's usually buffered by GFile / Python
    # / OS and written to disk in increments of the filesystem blocksize.
    # Complete write is guaranteed after Python has exited.
    shutil.rmtree(logdir)


def test_pytorch_summary(geometry_data, tmp_path):
    """Test writing summary from PyTorch"""
    torch = pytest.importorskip("torch")
    torch_tb = pytest.importorskip("torch.utils.tensorboard")
    SummaryWriter = torch_tb.SummaryWriter
    logdir = str(tmp_path)
    writer = SummaryWriter(logdir)

    rng = np.random.default_rng()
    tensor_converter = (torch.from_numpy, o3d.core.Tensor.from_numpy, np.array)

    cube, material = geometry_data['cube'], geometry_data['material']
    cube_custom_prop = geometry_data['cube_custom_prop']
    cube_ls, material_ls = geometry_data['cube_ls'], geometry_data[
        'material_ls']
    colors = geometry_data['colors']
    cube_labels = geometry_data['cube_labels']
    label_to_names = geometry_data['label_to_names']
    max_outputs = geometry_data['max_outputs']
    bboxes = geometry_data['bboxes']
    for step in range(3):
        cube[0].paint_uniform_color(colors[step][0])
        cube[1].paint_uniform_color(colors[step][1])
        cube_summary = to_dict_batch(cube)
        cube_summary.update(material)
        # Randomly convert to PyTorch, Open3D, Numpy tensors, or use property
        # reference
        if step > 0:
            cube_summary['vertex_positions'] = 0
            cube_summary['vertex_normals'] = 0
            cube_summary['vertex_colors'] = rng.choice(tensor_converter)(
                cube_summary['vertex_colors'])
        else:
            for prop, tensor in cube_summary.items():
                # skip material scalar and vector props
                if (not prop.startswith("material_") or
                        prop.startswith("material_texture_map_")):
                    cube_summary[prop] = rng.choice(tensor_converter)(tensor)
        writer.add_3d('cube', cube_summary, step=step, max_outputs=max_outputs)
        for key in tuple(cube_summary):  # Convert to PointCloud
            if key.startswith(('triangle_', 'material_texture_map_')):
                cube_summary.pop(key)
        cube_summary['vertex_custom'] = tuple(
            rng.choice(tensor_converter)(tensor)
            for tensor in cube_custom_prop[step])  # Add custom prop
        cube_summary['vertex_labels'] = tuple(
            rng.choice(tensor_converter)(tensor)
            for tensor in cube_labels[step])  # Add labels
        writer.add_3d('cube_pcd',
                      cube_summary,
                      step=step,
                      max_outputs=max_outputs,
                      label_to_names=label_to_names)
        cube_ls[0].paint_uniform_color(colors[step][0])
        cube_ls[1].paint_uniform_color(colors[step][1])
        cube_ls_summary = to_dict_batch(cube_ls)
        cube_ls_summary.update(material_ls)
        for prop, tensor in cube_ls_summary.items():
            if (not prop.startswith("material_") or
                    prop.startswith("material_texture_map_")):
                cube_ls_summary[prop] = rng.choice(tensor_converter)(tensor)
        writer.add_3d('cube_ls',
                      cube_ls_summary,
                      step=step,
                      max_outputs=max_outputs)
        if len(bboxes) > 0:
            writer.add_3d('bboxes', {'bboxes': bboxes[step]},
                          step=step,
                          logdir=logdir,
                          max_outputs=max_outputs,
                          label_to_names=label_to_names)

    sleep(0.25)  # msgpack writing disk flush time

    tags_ref = geometry_data['tags']
    dirpath_ref = [
        logdir,
        os.path.join(logdir, 'plugins'),
        os.path.join(logdir, 'plugins/Open3D')
    ]
    filenames_ref = geometry_data['filenames']
    dirpath, filenames = [], []
    for dp, unused_dn, fn in os.walk(logdir):
        dirpath.append(dp)
        filenames.append(fn)

    assert dirpath == dirpath_ref
    assert filenames[0][0].startswith(filenames_ref[0][0][:20])
    assert sorted(x.split('.')[0] for x in filenames[2]) == tags_ref
    assert all(fn.endswith('.msgpack') for fn in filenames[2])

    # Note: The event file written during this test cannot be reliably verified
    # in the same Python process, since it's usually buffered by GFile / Python
    # / OS and written to disk in increments of the filesystem blocksize.
    # Complete write is guaranteed after Python has exited.
    shutil.rmtree(logdir)


def check_material_dict(o3d_geo, material, batch_idx):
    assert o3d_geo.has_valid_material()
    assert o3d_geo.material.material_name == material['material_name'][
        batch_idx]
    for prop, value in material.items():
        if prop == "material_name":
            assert o3d_geo.material.material_name == material[prop][batch_idx]
        elif prop.startswith("material_scalar_"):
            assert o3d_geo.material.scalar_properties[
                prop[16:]] == value[batch_idx]
        elif prop.startswith("material_vector_"):
            assert all(o3d_geo.material.vector_properties[prop[16:]] ==
                       value[batch_idx])
        elif prop.startswith("material_texture_map_"):
            if value[batch_idx].dtype == np.uint8:
                ref_value = value[batch_idx]
            elif value[batch_idx].dtype == np.uint16:
                ref_value = (value[batch_idx] // 256).astype(np.uint8)
            elif value[batch_idx].dtype in (np.float32, np.float64):
                ref_value = (value[batch_idx] * 255).astype(np.uint8)
            else:
                raise ValueError("Reference texture map has unsupported dtype:"
                                 f"{value[batch_idx].dtype}")
            assert (o3d_geo.material.texture_maps[
                prop[21:]].as_tensor().numpy() == ref_value).all()


@pytest.fixture
def logdir():
    """Extract logdir zip to provide logdir for tests, cleanup afterwards."""
    shutil.unpack_archive(
        os.path.join(test_data_dir, "test_tensorboard_plugin.zip"))
    yield "test_tensorboard_plugin"
    shutil.rmtree("test_tensorboard_plugin")


def test_plugin_data_reader(geometry_data, logdir):
    """Test reading summary data"""
    cube, material = geometry_data['cube'], geometry_data['material']
    cube_custom_prop = geometry_data['cube_custom_prop']
    cube_ls, material_ls = geometry_data['cube_ls'], geometry_data[
        'material_ls']
    colors = geometry_data['colors']
    max_outputs = geometry_data['max_outputs']
    cube_labels = geometry_data['cube_labels']
    label_to_names_ref = geometry_data['label_to_names']
    bboxes_ref = geometry_data['bboxes']
    tags_ref = geometry_data['tags']

    reader = Open3DPluginDataReader(logdir)
    assert reader.is_active()
    assert reader.run_to_tags == {'.': tags_ref}
    assert reader.get_label_to_names('.', 'cube_pcd') == label_to_names_ref
    assert reader.get_label_to_names('.', 'bboxes') == label_to_names_ref
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
                                            step_to_idx)[0]
            assert (cube_out.vertex['positions'] == cube_ref.vertex['positions']
                   ).all()
            assert (
                cube_out.vertex['normals'] == cube_ref.vertex['normals']).all()
            assert (
                cube_out.vertex['colors'] == cube_ref.vertex['colors']).all()
            assert (cube_out.triangle['indices'] == cube_ref.triangle['indices']
                   ).all()
            check_material_dict(cube_out, material, batch_idx)

            cube_pcd_out = reader.read_geometry(".", "cube_pcd", step,
                                                batch_idx, step_to_idx)[0]
            assert (cube_pcd_out.point['positions'] ==
                    cube_ref.vertex['positions']).all()
            assert cube_pcd_out.has_valid_material()
            assert (cube_pcd_out.point['normals'] == cube_ref.vertex['normals']
                   ).all()
            assert (cube_pcd_out.point['colors'] == cube_ref.vertex['colors']
                   ).all()
            assert (cube_pcd_out.point['custom'].numpy() ==
                    cube_custom_prop[step][batch_idx]).all()
            assert (cube_pcd_out.point['labels'].numpy() == cube_labels[step]
                    [batch_idx]).all()
            for key in tuple(material):
                if key.startswith('material_texture_map_'):
                    material.pop(key)
            check_material_dict(cube_pcd_out, material, batch_idx)

            cube_ls[batch_idx].paint_uniform_color(colors[step][batch_idx])
            cube_ls_ref = o3d.t.geometry.LineSet.from_legacy(cube_ls[batch_idx])
            cube_ls_ref.line["indices"] = cube_ls_ref.line["indices"].to(
                o3d.core.int32)
            cube_ls_ref.line['colors'] = (cube_ls_ref.line['colors'] * 255).to(
                o3d.core.uint8)

            cube_ls_out = reader.read_geometry(".", "cube_ls", step, batch_idx,
                                               step_to_idx)[0]
            assert (cube_ls_out.point['positions'] ==
                    cube_ls_ref.point['positions']).all()
            assert (cube_ls_out.line['indices'] == cube_ls_ref.line['indices']
                   ).all()
            assert (
                cube_ls_out.line['colors'] == cube_ls_ref.line['colors']).all()
            check_material_dict(cube_ls_out, material_ls, batch_idx)

            bbox_ls_out, data_bbox_proto = reader.read_geometry(
                ".", "bboxes", step, batch_idx, step_to_idx)
            bbox_ls_ref = o3d.t.geometry.LineSet.from_legacy(
                BoundingBox3D.create_lines(bboxes_ref[step][batch_idx]))
            bbox_ls_ref.line["indices"] = bbox_ls_ref.line["indices"].to(
                o3d.core.int32)
            assert (bbox_ls_out.point["positions"] ==
                    bbox_ls_ref.point["positions"]).all()
            assert (bbox_ls_out.line["indices"] == bbox_ls_ref.line["indices"]
                   ).all()
            assert "colors" not in bbox_ls_out.line
            label_conf_ref = tuple((bb.label_class, bb.confidence)
                                   for bb in bboxes_ref[step][batch_idx])
            label_conf_out = tuple((bb.label, bb.confidence)
                                   for bb in data_bbox_proto.inference_result)
            np.testing.assert_allclose(label_conf_ref, label_conf_out)


@pytest.mark.skip(reason="This will only run on a machine with GPU and GUI.")
def test_tensorboard_app(logdir):
    with sp.Popen(['tensorboard', '--logdir', logdir]) as tb_proc:
        sleep(5)
        webbrowser.open('http://localhost:6006/')
        sleep(8)
        tb_proc.kill()
