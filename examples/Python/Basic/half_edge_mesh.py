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
    # Initialize a HalfEdgeTriangleMesh from TriangleMesh
    mesh = o3d.io.read_triangle_mesh("../../TestData/sphere.ply")
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox.min_bound = [-1, -1, -1]
    bbox.max_bound = [1, 0.6, 1]
    mesh = mesh.crop(bbox)
    het_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    draw_geometries_with_back_face([het_mesh])

    # Colorize boundary vertices to red
    vertex_colors = 0.75 * np.ones((len(het_mesh.vertices), 3))
    for boundary in het_mesh.get_boundaries():
        for vertex_id in boundary:
            vertex_colors[vertex_id] = [1, 0, 0]
    het_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    draw_geometries_with_back_face([het_mesh])
