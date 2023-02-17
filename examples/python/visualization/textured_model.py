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
        print("""Usage: texture-model.py [model directory]
    This example will load [model directory].obj plus any of albedo, normal,
    ao, metallic and roughness textures present.""")
        sys.exit()

    model_dir = sys.argv[1]
    model_name = os.path.join(model_dir, os.path.basename(model_dir) + ".obj")
    model = o3d.io.read_triangle_mesh(model_name)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    albedo_name = os.path.join(model_dir, "albedo.png")
    normal_name = os.path.join(model_dir, "normal.png")
    ao_name = os.path.join(model_dir, "ao.png")
    metallic_name = os.path.join(model_dir, "metallic.png")
    roughness_name = os.path.join(model_dir, "roughness.png")
    if os.path.exists(albedo_name):
        material.albedo_img = o3d.io.read_image(albedo_name)
    if os.path.exists(normal_name):
        material.normal_img = o3d.io.read_image(normal_name)
    if os.path.exists(ao_name):
        material.ao_img = o3d.io.read_image(ao_name)
    if os.path.exists(metallic_name):
        material.base_metallic = 1.0
        material.metallic_img = o3d.io.read_image(metallic_name)
    if os.path.exists(roughness_name):
        material.roughness_img = o3d.io.read_image(roughness_name)

    o3d.visualization.draw([{
        "name": "cube",
        "geometry": model,
        "material": material
    }])


if __name__ == "__main__":
    main()
