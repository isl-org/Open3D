# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/hidden_point_removal.py

import numpy as np
import open3d as o3d
import meshes


def mesh_generator():
    yield o3d.geometry.TriangleMesh.create_sphere()
    yield meshes.knot()
    yield meshes.bunny()
    yield meshes.armadillo()


if __name__ == "__main__":

    for mesh in mesh_generator():

        print("Convert mesh to a point cloud and estimate dimensions")
        pcl = o3d.geometry.PointCloud()
        pcl.points = mesh.vertices
        pcl.colors = mesh.vertex_colors
        diameter = np.linalg.norm(
            np.asarray(pcl.get_max_bound()) - np.asarray(pcl.get_min_bound()))

        print("Define parameters used for hidden_point_removal")
        camera = [diameter, diameter * 0.5, diameter * 0.5]
        radius = diameter * 100

        print("Create coordinate frame for visualizing the camera location")
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=diameter / 5, origin=camera)

        print("Remove all hidden points viewed from the camera location")
        hull, _ = pcl.hidden_point_removal(camera, radius)

        print("Visualize result")
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        pcl.paint_uniform_color((0.5, 0.5, 1))
        o3d.visualization.draw_geometries([pcl, hull_ls, camera_frame])

    print("Create a point cloud representing a sphere")
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    pcl = o3d.geometry.PointCloud()
    pcl.points = mesh.vertices

    print("Assign colors based on their index (green to red)")
    l = len(pcl.points)
    colors = np.array(
        [np.arange(0, l, 1) / l,
         np.arange(l, 0, -1) / l,
         np.zeros(l)]).transpose()
    pcl.colors = o3d.utility.Vector3dVector(colors)

    print("Remove all hidden points viewed from the camera location")
    mesh, pt_map = pcl.hidden_point_removal([4, 0, 0], 100)

    print("Add back colors using the point map")
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[np.asarray(pt_map)])

    print("Visualize the result")
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    o3d.visualization.draw_geometries([mesh, pcl])
