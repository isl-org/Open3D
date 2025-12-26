from initialize import *
from examples.python.Dayong.surface_fitting.align import *
from extrude import *
from examples.python.Dayong.surface_fitting.cut import *

def get_rotation_matrix(from_vec, to_vec):
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R

def create_plane_patch(center: np.ndarray,
                       width: float,
                       height: float,
                       thickness: float,
                       normal: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    创建一个中心为 center、有厚度的平面 patch（其实是薄盒子），朝向 normal。
    """
    plane = o3d.geometry.TriangleMesh.create_box(width, height, thickness)
    plane.translate(-plane.get_center())  # 移动到原点中心
    R = get_rotation_matrix(np.array([0, 0, 1]), normal)
    plane.rotate(R, center=(0, 0, 0))
    plane.translate(center)
    return plane

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    toy_path = os.path.join(base_dir, "STLs", "fish.stl")

    # Read and center badge mesh
    toy = read_stl_to_mesh(toy_path)
    toy.paint_uniform_color([1.0, 0.6, 0.3])  # orange
    # o3d.visualization.draw_geometries([toy], mesh_show_back_face=True)

    # 构造平面
    bbox = toy.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    plane = create_plane_patch(center=center,
                               width=extent[0] * 2,
                               height=extent[1] * 2,
                               thickness=3,
                               normal=np.array([0, 0, 1]))
    plane.paint_uniform_color([0.6, 0.8, 1.0])
    # o3d.visualization.draw_geometries([plane, toy], mesh_show_back_face=True)

    # Get o and its normal
    o = np.array([0, 0, 1])
    visualize_point_on_mesh(plane, o, 1.0)
    normal_o = np.array([0, 0, 1])

    p = select_o_point_from_ratio(toy, 0.5, 1, 0.5)  # fish背面
    visualize_point_on_mesh(toy, p, 0.1)
    normal_p = get_normal_at_point(toy, p)

    # Rotate toy
    R = get_rotation_matrix(from_vec=-normal_p, to_vec=normal_o) # 背面
    toy.rotate(R, center=p)  # rotate toy based on p

    # Move p to o
    offset = o - p
    toy.translate(offset)

    # # ray casting 投影：从平面外沿 -normal 方向投射到平面上
    # all_points = np.asarray(toy.vertices) + normal_o * 5.0  # 往法线方向偏移一段距离
    # projected_points, hit_mask = project_points_onto_mesh(all_points, -normal_o, plane)
    #
    # toy.vertices = o3d.utility.Vector3dVector(projected_points)
    # o3d.visualization.draw_geometries([plane, toy], mesh_show_back_face=True)
    #
    # thickness = 1
    # extruded_toy = extrude_shell(toy, thickness)
    # extruded_toy.paint_uniform_color([1.0, 0.6, 0.3])
    #
    # o3d.visualization.draw_geometries([extruded_toy], mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([plane, extruded_toy], mesh_show_back_face=True)