# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest

mi = pytest.importorskip("mitsuba")


@pytest.mark.parametrize("seam", [False, True])
def test_mitsuba_preserves_face_corner_attributes(seam):
    mi.set_variant("scalar_rgb")
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                         dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = positions[faces, :2].copy()
    if seam:
        uvs[1, 0] = [0.25, 0.5]
        uvs[1, 1] = [0.75, 0.5]
    normals = np.tile([0., 0., 1.], (4, 1)).astype(np.float32)
    colors = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(positions),
                                       o3d.core.Tensor(faces))
    mesh.vertex.normals = o3d.core.Tensor(normals)
    mesh.vertex.colors = o3d.core.Tensor(colors)
    mesh.triangle.texture_uvs = o3d.core.Tensor(uvs)
    mesh.material.set_default_properties()

    converted = mesh.to_mitsuba("square")
    parameters = mi.traverse(converted)
    out_faces = np.asarray(parameters["faces"]).reshape(-1, 3)
    out_uvs = np.asarray(parameters["vertex_texcoords"]).reshape(-1, 2)
    expected_uvs = uvs.copy()
    expected_uvs[..., 1] = 1 - expected_uvs[..., 1]
    np.testing.assert_allclose(out_uvs[out_faces], expected_uvs)
    for field, expected in [("vertex_positions", positions),
                            ("vertex_normals", normals),
                            ("vertex_color", colors)]:
        values = np.asarray(parameters[field]).reshape(-1, 3)
        np.testing.assert_allclose(values[out_faces], expected[faces])
    assert converted.vertex_count() == (6 if seam else 4)
    np.testing.assert_array_equal(mesh.triangle.indices.numpy(), faces)
    np.testing.assert_array_equal(mesh.triangle.texture_uvs.numpy(), uvs)
    np.testing.assert_array_equal(mesh.vertex.positions.numpy(), positions)


def test_mitsuba_untextured_mesh_keeps_indexed_vertices():
    mi.set_variant("scalar_rgb")
    mesh = o3d.t.geometry.TriangleMesh.create_box()
    converted = mesh.to_mitsuba("box", mi.load_dict({"type": "diffuse"}))
    parameters = mi.traverse(converted)
    np.testing.assert_array_equal(
        np.asarray(parameters["faces"]).reshape(-1, 3),
        mesh.triangle.indices.numpy())
    np.testing.assert_allclose(
        np.asarray(parameters["vertex_positions"]).reshape(-1, 3),
        mesh.vertex.positions.numpy())
