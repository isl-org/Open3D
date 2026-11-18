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


@pytest.mark.parametrize("translation", [[0, 0, 0], [10, -20, 30]])
def test_mitsuba_mesh_is_ready_for_rendering(translation):
    mi.set_variant("scalar_rgb")
    mesh = o3d.t.geometry.TriangleMesh.create_box(2, 3, 4)
    mesh.translate(translation)
    mesh.material.set_default_properties()

    converted = mesh.to_mitsuba("box")
    bounds = converted.bbox()
    assert bounds.valid()
    np.testing.assert_allclose(np.array(bounds.min), translation)
    np.testing.assert_allclose(np.array(bounds.max),
                               np.array(translation) + [2, 3, 4])
    assert converted.surface_area() == pytest.approx(52)

    origin = np.array(translation, dtype=float) + [1, 1, 10]
    scene = mi.load_dict({"type": "scene", "mesh": converted})
    interaction = scene.ray_intersect(mi.Ray3f(origin, [0, 0, -1]))
    assert interaction.is_valid()
    assert interaction.t == pytest.approx(6)


def test_mitsuba_mesh_is_ready_with_custom_bsdf():
    mi.set_variant("scalar_rgb")
    mesh = o3d.t.geometry.TriangleMesh.create_box()
    converted = mesh.to_mitsuba("box", mi.load_dict({"type": "diffuse"}))
    assert converted.bbox().valid()
    assert converted.surface_area() == pytest.approx(6)


def test_mitsuba_mesh_preserves_authored_shading_normals():
    mi.set_variant("scalar_rgb")
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
        o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.Int64))
    normals = np.tile([0., 0.6, 0.8], (3, 1)).astype(np.float32)
    mesh.vertex.normals = o3d.core.Tensor(normals)
    converted = mesh.to_mitsuba("triangle", mi.load_dict({"type": "diffuse"}))
    parameters = mi.traverse(converted)
    np.testing.assert_allclose(np.asarray(parameters["vertex_normals"]).reshape(
        -1, 3),
                               normals,
                               atol=1e-7)
    scene = mi.load_dict({"type": "scene", "mesh": converted})
    hit = scene.ray_intersect(mi.Ray3f([0.2, 0.2, 1.], [0., 0., -1.]))
    assert hit.is_valid()
    np.testing.assert_allclose(np.asarray(hit.sh_frame.n), [0., 0.6, 0.8],
                               atol=1e-7)
    np.testing.assert_array_equal(mesh.vertex.normals.numpy(), normals)
