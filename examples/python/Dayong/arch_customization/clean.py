import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --- Geometric (bounding-box) Y trim ---
def trim_by_y_range(
    pcd: o3d.geometry.PointCloud,
    y_min_ratio: float = 0.0,
    y_max_ratio: float = 1.0,
    return_indices: bool = True,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Geometric (bounding-box) trim along Y.

    Unlike percentile/quantile (which is based on point counts), this uses the Y extent
    of the point cloud's bounding box.

    Example:
        y_min_ratio=0.2, y_max_ratio=1.0  -> cut off the bottom 20% of the Y-range

    Args:
        pcd: Open3D PointCloud (already PCA-aligned)
        y_min_ratio: float in [0, 1]. 0.2 means start at 20% of Y-range from y_min.
        y_max_ratio: float in [0, 1]. 1.0 means up to y_max.
        return_indices: whether to return indices into input pcd

    Returns:
        trimmed_pcd: filtered point cloud
        idx: indices into input pcd (empty if return_indices=False)
    """
    if not (0.0 <= y_min_ratio <= y_max_ratio <= 1.0):
        raise ValueError("Ratios must satisfy 0 ≤ y_min_ratio ≤ y_max_ratio ≤ 1")

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return o3d.geometry.PointCloud(), np.array([], dtype=int)

    y = pts[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())

    # If all Y are identical, either keep all (if range includes that value) or none.
    if y_max <= y_min + 1e-12:
        idx = np.arange(len(y), dtype=int)
        trimmed_pcd = pcd.select_by_index(idx)
        return (trimmed_pcd, idx) if return_indices else (trimmed_pcd, np.array([], dtype=int))

    y_lo = y_min + y_min_ratio * (y_max - y_min)
    y_hi = y_min + y_max_ratio * (y_max - y_min)

    mask = (y >= y_lo) & (y <= y_hi)
    idx = np.where(mask)[0]

    trimmed_pcd = pcd.select_by_index(idx)

    if return_indices:
        return trimmed_pcd, idx
    else:
        return trimmed_pcd, np.array([], dtype=int)