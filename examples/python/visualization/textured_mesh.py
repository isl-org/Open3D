# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import sys
import os
import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print("""Usage: textured-mesh.py [model directory]
    This example will load [model directory].obj plus any of albedo, normal,
    ao, metallic and roughness textures present. The textures should be named
    albedo.png, normal.png, ao.png, metallic.png and roughness.png
    respectively.""")
        sys.exit()

    model_dir = os.path.normpath(os.path.realpath(sys.argv[1]))
    model_name = os.path.join(model_dir, os.path.basename(model_dir) + ".obj")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.io.read_triangle_mesh(model_name))
    material = mesh.material
    material.material_name = "defaultLit"

    names_to_o3dprop = {"ao": "ambient_occlusion"}
    for texture in ("albedo", "normal", "ao", "metallic", "roughness"):
        texture_file = os.path.join(model_dir, texture + ".png")
        if os.path.exists(texture_file):
            texture = names_to_o3dprop.get(texture, texture)
            material.texture_maps[texture] = o3d.t.io.read_image(texture_file)
    if "metallic" in material.texture_maps:
        material.scalar_properties["metallic"] = 1.0

    o3d.visualization.draw(mesh, title=model_name)


if __name__ == "__main__":
    main()
