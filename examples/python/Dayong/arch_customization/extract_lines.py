import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import math
from typing import Dict, Tuple, Optional, List
from matplotlib.path import Path

def extract_boundary_lines(
    pcd: o3d.geometry.PointCloud,
    percentage=96,
    return_indices: bool = False,
    base_indices: Optional[np.ndarray] = None,
):
    points = np.asarray(pcd.points)
    if base_indices is None:
        base_indices = np.arange(len(points), dtype=int)
    else:
        base_indices = np.asarray(base_indices, dtype=int)
        if len(base_indices) != len(points):
            raise ValueError("base_indices must have same length as pcd points")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    normals = np.asarray(pcd.normals)

    # Build KD-tree for neighborhood searches
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    cliff_scores = []
    for i in range(len(points)):
        # Find neighbors
        [k, idx, _] = kdtree.search_radius_vector_3d(points[i], 1.0)

        if k < 5:
            cliff_scores.append(0)
            continue

        neighbor_points = points[idx[1:], :]
        neighbor_normals = normals[idx[1:], :]

        # Compute 3 signals from the neighborhood：

        # Height variation (Z-direction)
        # High when the neighborhood has a “step”, “ridge”, or shape change
        height_std = np.std(neighbor_points[:, 2])

        # Normal variation (direction change)
        # High when normals change quickly around that point (creases/edges)
        normal_variation = 1 - np.abs(np.dot(normals[i], neighbor_normals.T).mean())

        # Vertical of normal (cliffs are steep).
        # vertical = abs(nz)
        # This term is meant to boost “cliff edges”, because on a cliff the normals are more horizontal,
        # so |nz| is smaller, making (1 - |nz|) larger.
        vertical = np.abs(normals[i][2])  # How horizontal the normal is

        # cliff_score = height_std * normal_variation * (1 - vertical)
        cliff_score = 1 - vertical
        cliff_scores.append(cliff_score)

    # Extract cliff points
    cliff_scores = np.array(cliff_scores)
    threshold = np.percentile(cliff_scores, percentage)
    cliff_indices = np.where(cliff_scores > threshold)[0]
    cliff_base_idx = base_indices[cliff_indices]

    # Visualize
    cliff_pcd = pcd.select_by_index(cliff_indices)
    # cliff_pcd.paint_uniform_color([1, 0, 0])  # Red for cliff edge
    o3d.visualization.draw_geometries([cliff_pcd.paint_uniform_color([1.0, 1.0, 0.0])], mesh_show_back_face=True, window_name="All Boundaries")
    if return_indices:
        return cliff_pcd, cliff_base_idx
    return cliff_pcd