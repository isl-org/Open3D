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
"""Demo scene demonstrating models, built-in shapes, and materials"""

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
        print("""This demo requires assets that appear to be missing.
Please execute the follow commands:
```
cd examples/test_data
wget https://github.com/isl-org/open3d_downloads/releases/download/o3d_demo_scene/demo_scene_assets.tgz
tar xzvf demo_scene_assets.tgz
cd ../..
python examples/python/visualization/demo_scene.py
```
""")
        exit(1)


def create_material(directory, name):
    '''
    Convenience function for creating material and loading a set of associated shaders
    '''
    mat = vis.Material('defaultLit')
    # Color, Roughness and Normal textures are always present
    mat.texture_maps['albedo'] = o3d.t.io.read_image(
        os.path.join(directory, name + '_Color.jpg'))
    mat.texture_maps['roughness'] = o3d.t.io.read_image(
        os.path.join(directory, name + '_Roughness.jpg'))
    mat.texture_maps['normal'] = o3d.t.io.read_image(
        os.path.join(directory, name + '_NormalDX.jpg'))
    # Ambient occlusion and metal textures are not always available
    # NOTE: Checking for their existence is not necessary but checking first
    # avoids annoying warning output
    ao_img_name = os.path.join(directory, name + '_AmbientOcclusion.jpg')
    metallic_img_name = os.path.join(directory, name + '_Metalness.jpg')
    if os.path.exists(ao_img_name):
        mat.texture_maps['ambient_occlusion'] = o3d.t.io.read_image(ao_img_name)
    if os.path.exists(metallic_img_name):
        mat.texture_maps['metallic'] = o3d.t.io.read_image(metallic_img_name)
        mat.scalar_properties['metallic'] = 1.0
    return mat


def convert_material_record(mat_record):
    mat = vis.Material('defaultLit')
    # Convert scalar paremeters
    mat.vector_properties['base_color'] = mat_record.base_color
    mat.scalar_properties['metallic'] = mat_record.base_metallic
    mat.scalar_properties['roughness'] = mat_record.base_roughness
    mat.scalar_properties['reflectance'] = mat_record.base_reflectance
    mat.texture_maps['albedo'] = o3d.t.geometry.Image.from_legacy(
        mat_record.albedo_img)
    mat.texture_maps['normal'] = o3d.t.geometry.Image.from_legacy(
        mat_record.normal_img)
    mat.texture_maps['ao_rough_metal'] = o3d.t.geometry.Image.from_legacy(
        mat_record.ao_rough_metal_img)
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
    a_cube = o3d.t.geometry.TriangleMesh.from_legacy(a_cube)

    a_sphere = o3d.geometry.TriangleMesh.create_sphere(2.5,
                                                       resolution=40,
                                                       create_uv_map=True)
    a_sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi / 2, 0, 0))
    a_sphere.rotate(rotate_90)
    a_sphere.translate((5, 2.4, 0))
    a_sphere = o3d.t.geometry.TriangleMesh.from_legacy(a_sphere)

    a_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        1.0, 4.0, 30, 4, True)
    a_cylinder.compute_triangle_normals()
    a_cylinder.rotate(rotate_90)
    a_cylinder.translate((10, 2, 0))
    a_cylinder = o3d.t.geometry.TriangleMesh.from_legacy(a_cylinder)

    a_ico = o3d.geometry.TriangleMesh.create_icosahedron(1.25,
                                                         create_uv_map=True)
    a_ico.compute_triangle_normals()
    a_ico.translate((-10, 2, 0))
    a_ico = o3d.t.geometry.TriangleMesh.from_legacy(a_ico)

    # Load an OBJ model for our scene
    helmet = o3d.io.read_triangle_model(
        "examples/test_data/demo_scene_assets/FlightHelmet.gltf")
    helmet_parts = []
    for m in helmet.meshes:
        # m.mesh.paint_uniform_color((1.0, 0.75, 0.3))
        m.mesh.scale(10.0, (0.0, 0.0, 0.0))
        helmet_parts.append(m)

    # Create a ground plane
    ground_plane = o3d.geometry.TriangleMesh.create_box(
        50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
    ground_plane.compute_triangle_normals()
    rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi, 0, 0))
    ground_plane.rotate(rotate_180)
    ground_plane.translate((-25.0, -0.1, -25.0))
    ground_plane.paint_uniform_color((1, 1, 1))
    ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)

    # Material to make ground plane more interesting - a rough piece of glass
    mat_ground = vis.Material("defaultLitSSR")
    mat_ground.scalar_properties['roughness'] = 0.15
    mat_ground.scalar_properties['reflectance'] = 0.72
    mat_ground.scalar_properties['transmission'] = 0.6
    mat_ground.scalar_properties['thickness'] = 0.3
    mat_ground.scalar_properties['absorption_distance'] = 0.1
    mat_ground.vector_properties['absorption_color'] = np.array(
        [0.82, 0.98, 0.972, 1.0])
    mat_ground.texture_maps['albedo'] = o3d.t.io.read_image(
        "examples/test_data/demo_scene_assets/PaintedPlaster017_Color.jpg")
    mat_ground.texture_maps['roughness'] = o3d.t.io.read_image(
        "examples/test_data/demo_scene_assets/noiseTexture.png")
    mat_ground.texture_maps['normal'] = o3d.t.io.read_image(
        "examples/test_data/demo_scene_assets/PaintedPlaster017_NormalDX.jpg")
    ground_plane.material = mat_ground

    # Load textures and create materials for each of our demo items
    a_cube.material = create_material("examples/test_data/demo_scene_assets",
                                      "WoodFloor050")
    a_sphere.material = create_material("examples/test_data/demo_scene_assets",
                                        "Tiles074")
    a_ico.material = create_material("examples/test_data/demo_scene_assets",
                                     "Terrazzo018")
    a_cylinder.material = create_material(
        "examples/test_data/demo_scene_assets", "Metal008")

    geoms = [{
        "name": "plane",
        "geometry": ground_plane
    }, {
        "name": "cube",
        "geometry": a_cube
    }, {
        "name": "cylinder",
        "geometry": a_cylinder
    }, {
        "name": "ico",
        "geometry": a_ico
    }, {
        "name": "sphere",
        "geometry": a_sphere
    }]
    # Load the helmet
    for part in helmet_parts:
        name = part.mesh_name
        tgeom = o3d.t.geometry.TriangleMesh.from_legacy(part.mesh)
        tgeom.material = convert_material_record(
            helmet.materials[part.material_idx])
        geoms.append({"name": name, "geometry": tgeom})
    return geoms


if __name__ == "__main__":
    check_for_required_assets()
    geoms = create_scene()
    vis.draw(geoms,
             bg_color=(0.8, 0.9, 0.9, 1.0),
             show_ui=True,
             width=1920,
             height=1080)
