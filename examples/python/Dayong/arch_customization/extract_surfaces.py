import open3d as o3d
import numpy as np


def extract_top(
    pcd: o3d.geometry.PointCloud,
    min_dot: float = 0.0
):
    """
    Split a point cloud into:
      - upper surface (normal · +Z >= min_dot)
      - rest (side + bottom)

    Args:
        pcd: Open3D PointCloud with normals
        min_dot:
            dot(n, +Z) threshold
            0.0  -> exactly 0–90 degrees
            0.2  -> ~78 degrees (recommended)
            0.4  -> ~66 degrees (very strict)

    Returns:
        upper_pcd, rest_pcd
    """
    if not pcd.has_normals():
        raise ValueError("PointCloud has no normals. Call estimate_normals() first.")

    normals = np.asarray(pcd.normals)
    nz = normals[:, 2]

    upper_mask = nz >= min_dot
    upper_idx = np.where(upper_mask)[0]
    rest_idx  = np.where(~upper_mask)[0]

    upper_pcd = pcd.select_by_index(upper_idx)
    rest_pcd  = pcd.select_by_index(rest_idx)

    return upper_pcd, rest_pcd

def extract_bottom(
    pcd: o3d.geometry.PointCloud,
    max_dot: float = -0.2 # 101.5° – 180°
):
    normals = np.asarray(pcd.normals)
    nz = normals[:, 2]

    bottom_idx = np.where(nz <= max_dot)[0]
    return pcd.select_by_index(bottom_idx)

def extract_side(
    pcd: o3d.geometry.PointCloud,
    abs_dot_max: float = 0.2 # 78.5° - 101.5°
):
    normals = np.asarray(pcd.normals)
    nz = normals[:, 2]

    side_idx = np.where(np.abs(nz) < abs_dot_max)[0]
    return pcd.select_by_index(side_idx)