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

import numpy as np
import open3d as o3d
import pytest


def test_set_mesh_data_deserialization():
    """Tests the deserialization of messages created with the set_mesh_data
    function.
    """

    def set_mesh_data_to_geometry(*args, **kwargs):
        bc = o3d.io.rpc.BufferConnection()
        o3d.io.rpc.set_mesh_data(*args, **kwargs, connection=bc)
        return o3d.io.rpc.data_buffer_to_meta_geometry(bc.get_buffer())

    rng = np.random

    # Geometry data
    verts = rng.rand(100, 3).astype(np.float32)
    tris = rng.randint(0, 100, size=[71, 3]).astype(np.int32)
    lines = rng.randint(0, 100, size=[82, 2]).astype(np.int32)

    dtypes = [np.uint8, np.int16, np.int32, np.float32, np.float64]
    vert_attrs = {
        'a':
            rng.uniform(0, 256, size=verts.shape).astype(rng.choice(dtypes)),
        'b':
            rng.uniform(0, 256,
                        size=verts.shape[:1] + (7,)).astype(rng.choice(dtypes)),
        'c':
            rng.uniform(0, 256,
                        size=verts.shape + (2,)).astype(rng.choice(dtypes)),
    }

    tri_attrs = {
        'a':
            rng.uniform(0, 256,
                        size=tris.shape[:1] + (1,)).astype(rng.choice(dtypes)),
        'b':
            rng.uniform(0, 256,
                        size=tris.shape[:1] + (3,)).astype(rng.choice(dtypes)),
        'c':
            rng.uniform(0, 256,
                        size=tris.shape[:1] + (5,)).astype(rng.choice(dtypes)),
    }

    line_attrs = {
        'a':
            rng.uniform(0, 256,
                        size=lines.shape[:1] + (1,)).astype(rng.choice(dtypes)),
        'b':
            rng.uniform(0, 256, size=lines.shape[:1] + (1, 2, 3)).astype(
                rng.choice(dtypes)),
        'c':
            rng.uniform(0, 256,
                        size=lines.shape[:1] + (2,)).astype(rng.choice(dtypes)),
    }

    # Material data
    material_name = "defaultUnlit"
    material_scalar_attributes = {'a': rng.uniform(0, 1)}
    material_vector_attributes = {'a': rng.uniform(0, 1, (4,))}
    texture_maps = {
        'a': rng.uniform(0, 256, size=(2, 2)).astype(rng.choice(dtypes)),
        'b': rng.uniform(0, 256, size=(2, 2, 1)).astype(rng.choice(dtypes)),
        'c': rng.uniform(0, 256, size=(2, 2, 3)).astype(rng.choice(dtypes)),
    }
    o3d_texture_maps = {
        key: o3d.t.geometry.Image(o3d.core.Tensor(value))
        for key, value in texture_maps.items()
    }

    # PointCloud
    path, time, geom = set_mesh_data_to_geometry(vertices=verts,
                                                 vertex_attributes=vert_attrs,
                                                 path="pcd",
                                                 time=123)
    assert path == "pcd"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.PointCloud)
    np.testing.assert_equal(geom.point['positions'].numpy(), verts)
    for key, value in vert_attrs.items():
        np.testing.assert_equal(geom.point[key].numpy(), value)

    # TriangleMesh
    path, time, geom = set_mesh_data_to_geometry(
        vertices=verts,
        faces=tris,
        vertex_attributes=vert_attrs,
        face_attributes=tri_attrs,
        material=material_name,
        material_scalar_attributes=material_scalar_attributes,
        material_vector_attributes=material_vector_attributes,
        texture_maps=o3d_texture_maps,
        path="trimesh",
        time=123)

    assert path == "trimesh"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.TriangleMesh)
    np.testing.assert_equal(geom.vertex['positions'].numpy(), verts)
    np.testing.assert_equal(geom.triangle['indices'].numpy(), tris)
    for key, value in vert_attrs.items():
        np.testing.assert_equal(geom.vertex[key].numpy(), value)
    for key, value in tri_attrs.items():
        np.testing.assert_equal(geom.triangle[key].numpy(), value)

    # Material test
    assert geom.material.material_name == material_name
    assert len(
        geom.material.scalar_properties) == len(material_scalar_attributes)
    for key, value in geom.material.scalar_properties.items():
        np.testing.assert_allclose(material_scalar_attributes[key], value)
    assert len(
        geom.material.vector_properties) == len(material_vector_attributes)
    for key, value in geom.material.vector_properties.items():
        np.testing.assert_allclose(material_vector_attributes[key], value)
    assert len(geom.material.texture_maps) == len(texture_maps)
    for key, value in geom.material.texture_maps.items():
        np.testing.assert_equal(np.squeeze(texture_maps[key]),
                                np.squeeze(value.as_tensor().numpy()))

    # Catch Material errors
    with pytest.raises(
            RuntimeError,
            match="SetMeshData: Please provide a material for the texture maps"
    ):
        path, time, geom = set_mesh_data_to_geometry(
            vertices=verts,
            faces=tris,
            material="",
            material_scalar_attributes=material_scalar_attributes,
            material_vector_attributes=material_vector_attributes,
            texture_maps=o3d_texture_maps,
            path="trimesh",
            time=123)

    # LineSet
    path, time, geom = set_mesh_data_to_geometry(vertices=verts,
                                                 lines=lines,
                                                 vertex_attributes=vert_attrs,
                                                 line_attributes=line_attrs,
                                                 path="lines",
                                                 time=123)
    assert path == "lines"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.LineSet)
    np.testing.assert_equal(geom.point['positions'].numpy(), verts)
    np.testing.assert_equal(geom.line['indices'].numpy(), lines)
    for key, value in vert_attrs.items():
        np.testing.assert_equal(geom.point[key].numpy(), value)
    for key, value in line_attrs.items():
        np.testing.assert_equal(geom.line[key].numpy(), value)

    #
    # Test partial data
    #
    path, time, geom = set_mesh_data_to_geometry(vertex_attributes=vert_attrs,
                                                 path="pcd",
                                                 time=123,
                                                 o3d_type="PointCloud")
    assert path == "pcd"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.PointCloud)
    for key, value in vert_attrs.items():
        np.testing.assert_equal(geom.point[key].numpy(), value)

    path, time, geom = set_mesh_data_to_geometry(vertex_attributes=vert_attrs,
                                                 face_attributes=tri_attrs,
                                                 path="trimesh",
                                                 time=123,
                                                 o3d_type="TriangleMesh")
    assert path == "trimesh"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.TriangleMesh)
    for key, value in vert_attrs.items():
        np.testing.assert_equal(geom.vertex[key].numpy(), value)
    for key, value in tri_attrs.items():
        np.testing.assert_equal(geom.triangle[key].numpy(), value)

    path, time, geom = set_mesh_data_to_geometry(vertex_attributes=vert_attrs,
                                                 line_attributes=line_attrs,
                                                 path="lines",
                                                 time=123,
                                                 o3d_type="LineSet")
    assert path == "lines"
    assert time == 123
    assert isinstance(geom, o3d.t.geometry.LineSet)
    for key, value in vert_attrs.items():
        np.testing.assert_equal(geom.point[key].numpy(), value)
    for key, value in line_attrs.items():
        np.testing.assert_equal(geom.line[key].numpy(), value)

    # Without o3d_type and no primary key data the returned object is None
    path, time, geom = set_mesh_data_to_geometry(vertex_attributes=vert_attrs,
                                                 path="unknown",
                                                 time=123,
                                                 o3d_type="")
    assert path == "unknown"
    assert time == 123
    assert geom is None


def test_recv_msgpack():
    """Test receiving messages constructed with msgpack.
    """
    msgpack = pytest.importorskip('msgpack')

    def numpy_to_Array(arr):
        if isinstance(arr, np.ndarray):
            return {
                'type': arr.dtype.str,
                'shape': arr.shape,
                'data': arr.tobytes()
            }
        raise ValueError('Object is not a Numpy array.')

    verts = np.array([[1, 2, 3]], dtype=np.float32)
    roughness = 0.3
    base_color = [0.2, 0.1, 0.9, 0.77]
    data = msgpack.packb({'msg_id': 'set_mesh_data'})
    data += msgpack.packb({
        'path': 'test',
        'data': {
            'vertices': numpy_to_Array(verts),
            'material': 'lit',
            'material_scalar_attributes': {
                'roughness': roughness
            },
            'material_vector_attributes': {
                'base_color': base_color
            }
        }
    })

    out_o3d = o3d.io.rpc.data_buffer_to_meta_geometry(data)
    assert out_o3d[:2] == ("test", 0.)
    assert np.allclose(out_o3d[2].point["positions"].numpy(), verts)
    assert np.isclose(out_o3d[2].material.scalar_properties['roughness'],
                      roughness)
    assert np.allclose(out_o3d[2].material.vector_properties['base_color'],
                       base_color)
