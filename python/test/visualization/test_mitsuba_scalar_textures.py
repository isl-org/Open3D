# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest
from open3d.visualization.to_mitsuba import o3d_material_to_bsdf

mi = pytest.importorskip("mitsuba")


@pytest.mark.parametrize("property_name,parameter_name",
                         [("roughness", "roughness"), ("metallic", "metallic"),
                          ("anisotropy", "anisotropic")])
@pytest.mark.parametrize("value", [64, 128, 192])
def test_mitsuba_scalar_textures_preserve_linear_values(property_name,
                                                        parameter_name, value):
    mi.set_variant("scalar_rgb")
    material = o3d.t.geometry.TriangleMesh().material
    material.set_default_properties()
    material.texture_maps[property_name] = o3d.t.geometry.Image(
        o3d.core.Tensor(np.full((2, 2, 3), value, dtype=np.uint8)))
    bsdf = o3d_material_to_bsdf(material)
    texture = np.asarray(mi.traverse(bsdf)[parameter_name + ".data"])
    np.testing.assert_allclose(texture, value / 255., atol=5e-4, rtol=0)


def test_mitsuba_albedo_retains_srgb_decoding():
    mi.set_variant("scalar_rgb")
    material = o3d.t.geometry.TriangleMesh().material
    material.set_default_properties()
    material.texture_maps["albedo"] = o3d.t.geometry.Image(
        o3d.core.Tensor(np.full((2, 2, 3), 128, dtype=np.uint8)))
    bsdf = o3d_material_to_bsdf(material)
    texture = np.asarray(mi.traverse(bsdf)["base_color.data"])
    expected = ((128 / 255. + 0.055) / 1.055)**2.4
    np.testing.assert_allclose(texture, expected, atol=5e-4, rtol=0)


@pytest.mark.parametrize("name", ["roughness", "metallic", "anisotropy"])
def test_mitsuba_scalar_texture_matches_constant_material(name):
    mi.set_variant("scalar_rgb")
    material = o3d.t.geometry.TriangleMesh().material
    material.set_default_properties()
    material.texture_maps[name] = o3d.t.geometry.Image(
        o3d.core.Tensor(np.full((2, 2, 3), 128, dtype=np.uint8)))
    textured = o3d_material_to_bsdf(material)
    constant = o3d.t.geometry.TriangleMesh().material
    constant.set_default_properties()
    constant.scalar_properties[name] = 128 / 255
    reference = o3d_material_to_bsdf(constant)

    interaction = mi.SurfaceInteraction3f()
    interaction.wi = [0, 0, 1]
    interaction.sh_frame = mi.Frame3f([0, 0, 1])
    interaction.uv = [0.5, 0.5]
    outgoing = mi.Vector3f([0.3, 0, np.sqrt(1 - 0.3**2)])
    context = mi.BSDFContext()
    np.testing.assert_allclose(
        np.array(textured.eval(context, interaction, outgoing)),
        np.array(reference.eval(context, interaction, outgoing)),
        rtol=1e-4)


def test_mitsuba_normal_map_remains_linear():
    mi.set_variant("scalar_rgb")
    material = o3d.t.geometry.TriangleMesh().material
    material.set_default_properties()
    material.texture_maps["normal"] = o3d.t.geometry.Image(
        o3d.core.Tensor(np.full((2, 2, 3), 128, dtype=np.uint8)))
    parameters = mi.traverse(o3d_material_to_bsdf(material))
    np.testing.assert_allclose(np.asarray(parameters["normalmap.data"]),
                               128 / 255,
                               atol=5e-4)
