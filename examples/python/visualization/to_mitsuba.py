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

import open3d as o3d
import mitsuba as mi


def render_mesh(mesh, mesh_center):
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {
            "type": "path"
        },
        "light": {
            "type": "envmap",
            "filename": "/home/renes/Downloads/solitude_interior_1k.exr",
            "scale": 1.5,
        },
        "sensor": {
            "type":
                "perspective",
            "focal_length":
                "50mm",
            "to_world":
                mi.ScalarTransform4f.look_at(origin=[0, 0, 5],
                                             target=mesh_center,
                                             up=[0, 1, 0]),
            "thefilm": {
                "type": "hdrfilm",
                "width": 1024,
                "height": 768,
            },
            "thesampler": {
                "type": "multijitter",
                "sample_count": 64,
            },
        },
        "themesh": mesh,
    })

    img = mi.render(scene, spp=256)
    return img


# Initialize mitsuba with LLVM variant which should be available on all
# platforms
mi.set_variant('llvm_ad_rgb')

# Load mesh using Open3D
dataset = o3d.data.MonkeyModel()
# NOTE: Once PR 5632 is merged we can use t IO directly here
mesh = o3d.io.read_triangle_mesh(dataset.path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
mesh_center = mesh.get_axis_aligned_bounding_box().get_center()

bsdf = mi.load_dict({
    "type": "principled",
    "base_color": {
        "type": "mesh_attribute",
        "name": "vertex_color",
    },
})

mi_mesh = o3d.visualization.to_mitsuba("monkey", mesh, bsdf=bsdf)
img = render_mesh(mi_mesh, mesh_center.numpy())
mi.Bitmap(img).write('test.exr')
