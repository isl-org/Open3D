# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/half_edge_mesh.py

import numpy as np
import open3d as o3d


def draw_geometries_with_back_face(geometries):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    render_option = visualizer.get_render_option()
    render_option.mesh_show_back_face = True
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh("../../TestData/sphere.ply")
    mesh = mesh.crop([-1, -1, -1], [1, 0.6, 1])
    # mesh.purge()
    mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_mesh(mesh)
    mesh.compute_vertex_normals()
    num_vertices = len(mesh.vertices)
    draw_geometries_with_back_face([mesh])

    # Find a boundary vertex
    boundaries = mesh.get_boundaries()
    assert len(boundaries) == 1
    boundary_vertices = boundaries[0]

    # Colorize boundary vertices
    vertex_colors = 0.75 * np.ones((num_vertices, 3))
    vertex_colors[boundary_vertices, :] = np.array([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    draw_geometries_with_back_face([mesh])
