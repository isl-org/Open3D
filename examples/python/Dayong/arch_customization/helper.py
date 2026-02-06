import numpy as np
import open3d as o3d

def normalize_mesh_to_mm(mesh: o3d.geometry.TriangleMesh, print_log=True):
    """
    Normalize mesh units to millimeters (m -> mm).
    - If bbox max extent < 1.0  -> assume meters, scale * 1000
    - Else keep as-is
    """
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())  # (dx, dy, dz)
    max_extent = float(np.max(extent))

    if print_log:
        print(f"[normalize_mesh_to_mm] extent = {extent}, max = {max_extent}")

    if max_extent < 1.0:  # likely meters
        mesh.scale(1000.0, center=mesh.get_center())
        if print_log:
            bbox2 = mesh.get_axis_aligned_bounding_box()
            extent2 = np.asarray(bbox2.get_extent())
            print(f"[normalize_mesh_to_mm] scaled to mm. new extent = {extent2}")
    return mesh

# -----------------------------
# Translation matrix helper
# -----------------------------

def get_translation_matrix(t: np.ndarray) -> np.ndarray:
    """Create a 4x4 translation matrix."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def align_min_x(target: o3d.geometry.PointCloud,
                source: o3d.geometry.PointCloud,
                gap: float = 0.0):
    """
    Move target in +X/-X so that:
        target.min_x == source.min_x + gap

    Returns:
        target: modified in-place point cloud
        T: 4x4 translation matrix applied to target

    Note:
        - source does NOT move
        - target is modified in-place
    """
    t_min = target.get_min_bound()
    s_min = source.get_min_bound()

    dx = (s_min[0] + gap) - t_min[0]
    t = np.array([dx, 0.0, 0.0], dtype=float)

    target.translate(t)
    return target, get_translation_matrix(t)


def align_min_y(target: o3d.geometry.PointCloud,
                source: o3d.geometry.PointCloud,
                gap: float = 0.0):
    """
    Move target in +Y/-Y so that:
        target.min_y == source.min_y + gap

    Returns:
        target: modified in-place point cloud
        T: 4x4 translation matrix applied to target

    Note:
        - source does NOT move
        - target is modified in-place
    """
    t_min = target.get_min_bound()
    s_min = source.get_min_bound()

    dy = (s_min[1] + gap) - t_min[1]
    t = np.array([0.0, dy, 0.0], dtype=float)

    target.translate(t)
    return target, get_translation_matrix(t)


def align_z(target_insole: o3d.geometry.PointCloud,
            source_foot: o3d.geometry.PointCloud,
            gap_mm: float = 2.0):
    """
    Z alignment for your case (IMPORTANT):
        foot bottom (source_foot.min_z)
        aligns to
        insole top surface (target_insole.max_z)
    with a gap:
        after move: target_insole.max_z == source_foot.min_z + gap_mm

    Why:
        - foot "bottom" is typically the lowest Z (min_z)
        - insole "top" is typically the highest Z (max_z) after you flip it correctly

    Returns:
        target_insole: modified in-place point cloud
        T: 4x4 translation matrix applied to target_insole

    Note:
        - source_foot does NOT move
        - target_insole is modified in-place
    """
    foot_min_z = source_foot.get_min_bound()[2]
    insole_max_z = target_insole.get_max_bound()[2]

    dz = (foot_min_z + gap_mm) - insole_max_z
    t = np.array([0.0, 0.0, dz], dtype=float)

    target_insole.translate(t)
    return target_insole, get_translation_matrix(t)

def get_translation_matrix_from_rotation(R: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 transform that rotates by R about a world-space center.
    """
    center = np.asarray(center).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    # x' = R (x - c) + c = R x + (c - R c)
    T[:3, 3] = center - R @ center
    return T

# Flip point cloud by 180 degrees around given axis (x/y/z)
def flip_point_cloud_180(pcd, axis='x'):
    if axis == 'x':
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
    elif axis == 'y':
        R = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
        ])
    elif axis == 'z':
        R = np.array([
            [-1, 0, 0],
            [ 0,-1, 0],
            [ 0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    c = pcd.get_center()  # 注意：这是“当前 pcd”中心
    t_flip = get_translation_matrix_from_rotation(R, c)
    pcd.transform(t_flip)
    return pcd, t_flip

def visualize_extracted_pcd(
    original_pcd: o3d.geometry.PointCloud,
    kept_idx: np.ndarray,
    window_name: str = "Y-percentile trim",
):
    """
    Gray = removed
    Red  = kept
    """
    vis = o3d.geometry.PointCloud(original_pcd)
    n = len(vis.points)

    colors = np.full((n, 3), 0.6)  # gray
    colors[kept_idx] = np.array([1.0, 0.0, 0.0])  # red

    vis.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [vis],
        window_name=window_name,
        width=1400,
        height=900,
        mesh_show_back_face=True,
    )

def pcd_to_mesh_bpa(
    arch_pcd: o3d.geometry.PointCloud,
    normal_radius: float = 6.0,
    normal_max_nn: int = 30,
    bpa_radii_mm=(2.0, 3.0, 4.0),
    simplify_target_tris: int = 80000,
    do_clean: bool = True
) -> o3d.geometry.TriangleMesh:
    """
    Convert a thin/sheet-like point cloud (arch) into a sheet mesh using BPA.
    This keeps it as a surface (no thickness).

    Args:
        arch_pcd: Open3D PointCloud (already extracted arch).
        normal_radius: radius (mm) for normal estimation.
        normal_max_nn: max neighbors for normal estimation.
        bpa_radii_mm: tuple/list of radii (mm) for BPA; usually 2~4mm works if your spacing ~0.5~1mm.
        simplify_target_tris: optional simplification target.
        do_clean: remove degenerate/duplicated triangles and non-manifold edges.

    Returns:
        TriangleMesh representing a thin sheet surface.
    """
    pcd = o3d.geometry.PointCloud(arch_pcd)

    # 1) Normals (required for BPA)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    # Make normals consistent (helps avoid flipped patches)
    try:
        pcd.orient_normals_consistent_tangent_plane(k=50)
    except Exception:
        pass

    # 2) BPA surface reconstruction (sheet-like)
    radii = o3d.utility.DoubleVector(list(bpa_radii_mm))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    # 3) Optional cleanup
    if do_clean:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    # 4) Optional simplification (keeps shape; reduces triangles)
    if simplify_target_tris is not None and len(mesh.triangles) > simplify_target_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_target_tris)

    mesh.compute_vertex_normals()
    return mesh