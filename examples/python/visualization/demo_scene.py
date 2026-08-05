# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Demo scene demonstrating models, built-in shapes, and materials"""

import numpy as np
import open3d as o3d
import open3d.visualization as vis


def create_scene():
    """Creates the geometry and materials for the demo scene and returns a
    dictionary suitable for draw call
    """
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
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
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
    helmet_data = o3d.data.FlightHelmetModel()
    helmet = o3d.io.read_triangle_model(helmet_data.path)
    helmet_parts = o3d.t.geometry.TriangleMesh.from_triangle_mesh_model(helmet)

    # Create a ground plane
    ground_plane = o3d.geometry.TriangleMesh.create_box(
        50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
    ground_plane.compute_triangle_normals()
    rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi, 0, 0))
    ground_plane.rotate(rotate_180)
    ground_plane.translate((-25.0, -0.1, -25.0))
    ground_plane.paint_uniform_color((1, 1, 1))
    ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)

    # Material to make ground plane more interesting - a rough piece of glass
    ground_plane.material = vis.Material("defaultLitSSR")
    ground_plane.material.scalar_properties['roughness'] = 0.15
    ground_plane.material.scalar_properties['reflectance'] = 0.72
    ground_plane.material.scalar_properties['transmission'] = 0.6
    ground_plane.material.scalar_properties['thickness'] = 0.3
    ground_plane.material.scalar_properties['absorption_distance'] = 0.1
    ground_plane.material.vector_properties['absorption_color'] = np.array(
        [0.82, 0.98, 0.972, 1.0])
    painted_plaster_texture_data = o3d.data.PaintedPlasterTexture()
    ground_plane.material.texture_maps['albedo'] = o3d.t.io.read_image(
        painted_plaster_texture_data.albedo_texture_path)
    ground_plane.material.texture_maps['normal'] = o3d.t.io.read_image(
        painted_plaster_texture_data.normal_texture_path)
    ground_plane.material.texture_maps['roughness'] = o3d.t.io.read_image(
        painted_plaster_texture_data.roughness_texture_path)

    # Load textures and create materials for each of our demo items
    wood_floor_texture_data = o3d.data.WoodFloorTexture()
    a_cube.material = vis.Material('defaultLit')
    a_cube.material.texture_maps['albedo'] = o3d.t.io.read_image(
        wood_floor_texture_data.albedo_texture_path)
    a_cube.material.texture_maps['normal'] = o3d.t.io.read_image(
        wood_floor_texture_data.normal_texture_path)
    a_cube.material.texture_maps['roughness'] = o3d.t.io.read_image(
        wood_floor_texture_data.roughness_texture_path)

    tiles_texture_data = o3d.data.TilesTexture()
    a_sphere.material = vis.Material('defaultLit')
    a_sphere.material.texture_maps['albedo'] = o3d.t.io.read_image(
        tiles_texture_data.albedo_texture_path)
    a_sphere.material.texture_maps['normal'] = o3d.t.io.read_image(
        tiles_texture_data.normal_texture_path)
    a_sphere.material.texture_maps['roughness'] = o3d.t.io.read_image(
        tiles_texture_data.roughness_texture_path)

    terrazzo_texture_data = o3d.data.TerrazzoTexture()
    a_ico.material = vis.Material('defaultLit')
    a_ico.material.texture_maps['albedo'] = o3d.t.io.read_image(
        terrazzo_texture_data.albedo_texture_path)
    a_ico.material.texture_maps['normal'] = o3d.t.io.read_image(
        terrazzo_texture_data.normal_texture_path)
    a_ico.material.texture_maps['roughness'] = o3d.t.io.read_image(
        terrazzo_texture_data.roughness_texture_path)

    metal_texture_data = o3d.data.MetalTexture()
    a_cylinder.material = vis.Material('defaultLit')
    a_cylinder.material.texture_maps['albedo'] = o3d.t.io.read_image(
        metal_texture_data.albedo_texture_path)
    a_cylinder.material.texture_maps['normal'] = o3d.t.io.read_image(
        metal_texture_data.normal_texture_path)
    a_cylinder.material.texture_maps['roughness'] = o3d.t.io.read_image(
        metal_texture_data.roughness_texture_path)
    a_cylinder.material.texture_maps['metallic'] = o3d.t.io.read_image(
        metal_texture_data.metallic_texture_path)

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
    for name, tmesh in helmet_parts.items():
        geoms.append({
            "name": name,
            "geometry": tmesh.scale(10.0, (0.0, 0.0, 0.0))
        })
    return geoms


if __name__ == "__main__":
    geoms = create_scene()
    vis.draw(geoms,
             bg_color=(0.8, 0.9, 0.9, 1.0),
             show_ui=True,
             width=1920,
             height=1080)
