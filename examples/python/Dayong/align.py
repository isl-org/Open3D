import numpy as np
import os
import open3d as o3d

def select_o_point_from_ratio(mesh: o3d.geometry.TriangleMesh,
                              x_ratio: float = 0.5,
                              y_ratio: float = 0.3,
                              z_ratio: float = 0.9) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)

    # Get the bounding box
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)

    # Calculate the ideal point that satisfies the xyz ratios
    target = min_bounds + (max_bounds - min_bounds) * np.array([x_ratio, y_ratio, z_ratio])

    # get the real point in the mesh that is closest to the ideal point
    distances = np.linalg.norm(vertices - target, axis=1)
    point = vertices[np.argmin(distances)]

    print("Selected Point: ", point)
    return point

def visualize_point_on_mesh(mesh: o3d.geometry.TriangleMesh, point: np.ndarray, radius=1.0):
    point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    point_sphere.translate(point)
    point_sphere.paint_uniform_color([1.0, 0, 0])  # Red
    o3d.visualization.draw_geometries([mesh, point_sphere], mesh_show_back_face=True)
    return None

def get_normal_at_point(mesh: o3d.geometry.TriangleMesh, point: np.ndarray) -> np.ndarray:
    """
    Calculate the normal of a point on the mesh
    """
    # 1. Sample point clouds from the mesh
    pcd = mesh.sample_points_uniformly(number_of_points=50000)
    pcd.estimate_normals()

    # 2. Get the nearest point from KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    _, idx, _ = kdtree.search_knn_vector_3d(point, 1)

    # 3. Get the normal for that point
    normal = np.asarray(pcd.normals)[idx[0]]
    print("The normal for the point is: ", normal)

    # Optional: visualize the normal
    normal_line_points = [point + normal * i for i in range(1, 50)]
    normal_spheres = []
    for p in normal_line_points:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        s.translate(p)
        s.paint_uniform_color([1.0, 0, 0])  # 红色
        normal_spheres.append(s)
    o3d.visualization.draw_geometries([mesh, *normal_spheres], mesh_show_back_face=True)

    return normal

def get_rotation_matrix(from_vec, to_vec):
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R