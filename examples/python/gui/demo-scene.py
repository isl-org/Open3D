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

#
# Demo scene demonstrating models, built-in shapes, and materials
#

# Use t geometry
# Use new Material

import math
import numpy as np
import os
import open3d as o3d
import open3d.visualization as vis


# Check for assets
def check_for_required_assets():
    '''
    Check for demo scene assets and print usage if necessary
    '''
    if not os.path.exists("examples/test_data/demo_scene_assets"):
        print(
            "This demo requires assets that appear to be missing. Please execute the follow:\n  cd examples/test_data\n  wget https://github.com/isl-org/open3d_downloads/releases/download/o3d_demo_scene/demo_scene_assets.tgz\n  tar xzvf demo_scene_assets.tgz\n  cd ../..\n  python examples/python/gui/demo-scene.py\n"
        )
        exit()


def create_material(directory, name):
    '''
    Convenience function for creating material and loading a set of associated shaders
    '''
    mat = vis.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    # Color, Roughness and Normal textures are always present
    mat.albedo_img = o3d.io.read_image(
        os.path.join(directory, name + '_Color.jpg'))
    mat.roughness_img = o3d.io.read_image(
        os.path.join(directory, name + '_Roughness.jpg'))
    mat.normal_img = o3d.io.read_image(
        os.path.join(directory, name + '_NormalDX.jpg'))
    # Ambient occlusion and metal textures are not always available
    # NOTE: Checking for their existence is not necessary but checking first
    # avoids annoying warning output
    ao_img_name = os.path.join(directory, name + '_AmbientOcclusion.jpg')
    metallic_img_name = os.path.join(directory, name + '_Metalness.jpg')
    if os.path.exists(ao_img_name):
        mat.ao_img = o3d.io.read_image(ao_img_name)
    if os.path.exists(metallic_img_name):
        mat.metallic_img = o3d.io.read_image(metallic_img_name)
        mat.base_metallic = 1.0
    return mat


def create_scene():
    '''
    Creates the geometry and materials for the demo scene and returns a dictionary suitable for draw call
    '''
    # Create some shapes for our scene
    a_cube = o3d.geometry.TriangleMesh.create_box(2,
                                                  4,
                                                  4,
                                                  create_uv_map=True,
                                                  map_texture_to_each_face=True)
    a_cube.compute_triangle_normals()
    a_cube.translate((-5, 0, -2))

    a_sphere = o3d.geometry.TriangleMesh.create_sphere(2.5,
                                                       resolution=40,
                                                       create_uv_map=True)
    a_sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi / 2, 0, 0))
    a_sphere.rotate(rotate_90)
    a_sphere.translate((5, 2.4, 0))

    a_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        1.0, 4.0, 30, 4, True)
    a_cylinder.compute_triangle_normals()
    a_cylinder.rotate(rotate_90)
    a_cylinder.translate((10, 2, 0))

    a_ico = o3d.geometry.TriangleMesh.create_icosahedron(1.25,
                                                         create_uv_map=True)
    a_ico.compute_triangle_normals()
    a_ico.translate((-10, 2, 0))

    # Load an OBJ model for our scene
    monkey = o3d.io.read_triangle_mesh("examples/test_data/monkey/monkey.obj")
    monkey.paint_uniform_color((1.0, 0.75, 0.3))
    monkey.scale(1.5, (0.0, 0.0, 0.0))
    monkey.translate((0, 1.4, 0))

    # Create a ground plane
    ground_plane = o3d.geometry.TriangleMesh.create_box(
        50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
    ground_plane.compute_triangle_normals()
    rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi, 0, 0))
    ground_plane.rotate(rotate_180)
    ground_plane.translate((-25.0, -0.1, -25.0))
    ground_plane.paint_uniform_color((1, 1, 1))

    # Material to make ground plane more interesting - a rough piece of glass
    mat_ground = vis.rendering.MaterialRecord()
    mat_ground.shader = "defaultLitSSR"
    mat_ground.base_roughness = 0.15
    mat_ground.base_reflectance = 0.72
    mat_ground.transmission = 0.6
    mat_ground.thickness = 0.3
    mat_ground.absorption_distance = 0.1
    mat_ground.absorption_color = np.array([0.82, 0.98, 0.972])
    mat_ground.albedo_img = o3d.io.read_image(
        "examples/test_data/demo_scene_assets/PaintedPlaster017_4K_Color.jpg")
    mat_ground.roughness_img = o3d.io.read_image(
        "examples/test_data/demo_scene_assets/noiseTexture.png")
    mat_ground.normal_img = o3d.io.read_image(
        "examples/test_data/demo_scene_assets/PaintedPlaster017_4K_NormalDX.jpg"
    )

    # Load textures and create materials for each of our demo items
    mat_monkey = create_material("examples/test_data/demo_scene_assets",
                                 "WoodFloor050_4K")
    mat_cube = create_material("examples/test_data/demo_scene_assets",
                               "Wood049_4K")
    mat_sphere = create_material("examples/test_data/demo_scene_assets",
                                 "Tiles074_4K")
    mat_ico = create_material("examples/test_data/demo_scene_assets",
                              "Terrazzo018_4K")
    mat_cylinder = create_material("examples/test_data/demo_scene_assets",
                                   "Metal008_2K")
    geoms = [{
        "name": "plane",
        "geometry": ground_plane,
        "material": mat_ground
    }, {
        "name": "cube",
        "geometry": a_cube,
        "material": mat_cube
    }, {
        "name": "monkey",
        "geometry": monkey,
        "material": mat_monkey
    }, {
        "name": "cylinder",
        "geometry": a_cylinder,
        "material": mat_cylinder
    }, {
        "name": "ico",
        "geometry": a_ico,
        "material": mat_ico
    }, {
        "name": "sphere",
        "geometry": a_sphere,
        "material": mat_sphere
    }]
    return geoms


if __name__ == "__main__":
    check_for_required_assets()
    geoms = create_scene()
    vis.draw(geoms,
             bg_color=(0.8, 0.9, 0.9, 1.0),
             show_ui=True,
             width=1920,
             height=1080,
             ibl="default",
             ibl_intensity=37500)
