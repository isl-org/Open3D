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
