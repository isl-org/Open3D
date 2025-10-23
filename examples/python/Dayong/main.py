import numpy as np
import os
import open3d as o3d
from open3d.visualization import gui, rendering

def read_stl_to_mesh(stl_path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    initialize_mesh(mesh)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

def initialize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # Translate the center of the mesh to the origin
    mesh.translate(-mesh.get_center())
    return mesh

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

def project_points_onto_mesh(points: np.ndarray,
                             direction: np.ndarray,
                             target_mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    使用 ray casting 将多个点沿着指定方向投射到 target_mesh 上。
    :param points: (N, 3) numpy array, 每个点是射线起点
    :param direction: (3,) numpy array, 所有射线的统一方向
    :param target_mesh: 被射线投射的目标 TriangleMesh
    :return: 命中的新点 (N, 3)，若未命中则保留原点
    """
    # 转为 tensor mesh
    target_t = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(target_t)

    # 构造射线数组 [N, 6] -> 每行 [x, y, z, dx, dy, dz]
    rays_np = np.hstack((points, np.tile(direction, (points.shape[0], 1)))).astype(np.float32)
    rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)

    # 执行 ray cast
    ans = scene.cast_rays(rays)
    t_hits = ans['t_hit'].numpy()

    # 命中的点：origin + t * direction
    hit_mask = np.isfinite(t_hits)
    projected_points = np.copy(points)
    projected_points[hit_mask] += direction * t_hits[hit_mask][:, np.newaxis]

    return projected_points


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    shoe_path = os.path.join(base_dir, "STLs", "shoe.stl")
    toy_path = os.path.join(base_dir, "STLs", "badge.stl")

    # Generate mesh for each stl
    shoe = read_stl_to_mesh(shoe_path)
    toy = read_stl_to_mesh(toy_path)
    shoe.paint_uniform_color([0.6, 0.8, 1.0]) # light blue
    toy.paint_uniform_color([1.0, 0.6, 0.3]) # orange
    toy = toy.scale(5.0, center=toy.get_center()) # scale to 2x size; temporary operation
    o3d.visualization.draw_geometries([shoe, toy], mesh_show_back_face=True)

    # Get o and its normal
    o = select_o_point_from_ratio(shoe, 0.5, 0.3, 0.9)
    visualize_point_on_mesh(shoe, o, 1.0)
    normal_o = get_normal_at_point(shoe, o)

    # Get p and its normal
    p = select_o_point_from_ratio(toy, 0.5, 0.5, 0)
    visualize_point_on_mesh(toy, p, 0.1)
    normal_p = get_normal_at_point(toy, p)

    # Rotate toy
    R = get_rotation_matrix(from_vec=-normal_p, to_vec=normal_o)
    toy.rotate(R, center=p)  # rotate toy based on p

    # move p to o
    offset = o - p
    toy.translate(offset)

    # Project the entire toy to the shoe
    all_points = np.asarray(toy.vertices)
    projected = project_points_onto_mesh(all_points, -normal_o, shoe)
    toy.vertices = o3d.utility.Vector3dVector(projected)

    o3d.visualization.draw_geometries([shoe, toy], mesh_show_back_face=True)