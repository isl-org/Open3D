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

def estimate_xy_spacing(points_xy: np.ndarray, k: int = 8) -> float:
    """
    Estimate typical spacing of points in XY using median distance to k-th neighbor.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    n = len(points_xy)
    if n < k + 1:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(points_xy)
    dists, _ = nbrs.kneighbors(points_xy)
    kth = dists[:, -1]  # distance to k-th neighbor (skip self)
    return float(np.median(kth))

def keep_largest_xy_component(
    pcd: o3d.geometry.PointCloud,
    radius: float = None,
    radius_scale: float = 2.5,
    k: int = 8,
    min_component_size: int = 30,
    return_indices: bool = True,
):
    """
    Treat points as a graph in XY: connect i<->j if dist_xy(i,j) <= radius,
    then keep the largest connected component.

    Parameters
    ----------
    pcd : PointCloud
        Input (e.g. pre_cut_line).
    radius : float
        Connection radius in XY. If None, auto-estimated from point spacing.
    radius_scale : float
        If radius is None, radius = radius_scale * median_kNN_distance.
        Increase if your pre-cut line is broken into segments (e.g. 3.0~4.0).
    k : int
        For spacing estimation (median distance to k-th neighbor).
    min_component_size : int
        Ignore components smaller than this as obvious outliers.
    return_indices : bool
        If True, returns (pcd_kept, kept_local_idx). kept_local_idx is indices into pcd.

    Returns
    -------
    kept_pcd : PointCloud
    kept_local_idx : np.ndarray (optional)
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        if return_indices:
            return o3d.geometry.PointCloud(), np.array([], dtype=int)
        return o3d.geometry.PointCloud()

    xy = pts[:, :2]

    if radius is None:
        spacing = estimate_xy_spacing(xy, k=k)
        if spacing <= 0:
            # fallback: very small cloud
            radius = 1.0
        else:
            radius = float(radius_scale * spacing)

    # Build neighbor graph using radius search
    nbrs = NearestNeighbors(radius=radius, algorithm="kd_tree").fit(xy)
    radius_neighbors = nbrs.radius_neighbors(xy, return_distance=False)

    rows = []
    cols = []
    for i, neigh in enumerate(radius_neighbors):
        for j in neigh:
            if i == j:
                continue
            rows.append(i)
            cols.append(j)

    n = len(xy)
    if len(rows) == 0:
        # No edges -> nothing connected; return as-is
        kept_local_idx = np.arange(n, dtype=int)
        kept_pcd = pcd.select_by_index(kept_local_idx)
        return (kept_pcd, kept_local_idx) if return_indices else kept_pcd

    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(csgraph=graph, directed=False)

    # Count component sizes
    sizes = np.bincount(labels, minlength=n_comp)

    # Filter out tiny components first
    valid = np.where(sizes >= min_component_size)[0]
    if len(valid) == 0:
        # if everything is tiny, just keep the biggest anyway
        best = int(np.argmax(sizes))
    else:
        best = int(valid[np.argmax(sizes[valid])])

    kept_local_idx = np.where(labels == best)[0].astype(int)
    kept_pcd = pcd.select_by_index(kept_local_idx)

    if return_indices:
        return kept_pcd, kept_local_idx
    return kept_pcd

def keep_component_with_largest_y_span(
    pcd: o3d.geometry.PointCloud,
    radius: float = None,
    radius_scale: float = 2.5,
    k: int = 8,
    min_component_size: int = 30,
    return_indices: bool = True,
):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        if return_indices:
            return o3d.geometry.PointCloud(), np.array([], dtype=int)
        return o3d.geometry.PointCloud()

    xy = pts[:, :2]
    if radius is None:
        spacing = estimate_xy_spacing(xy, k=k)
        radius = float(radius_scale * spacing) if spacing > 0 else 1.0

    nbrs = NearestNeighbors(radius=radius, algorithm="kd_tree").fit(xy)
    radius_neighbors = nbrs.radius_neighbors(xy, return_distance=False)

    rows, cols = [], []
    for i, neigh in enumerate(radius_neighbors):
        for j in neigh:
            if i != j:
                rows.append(i)
                cols.append(j)

    n = len(xy)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(csgraph=graph, directed=False)

    best_label = None
    best_span = -1.0

    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if len(idx) < min_component_size:
            continue
        y = pts[idx, 1]
        span = float(y.max() - y.min())
        if span > best_span:
            best_span = span
            best_label = c

    if best_label is None:
        best_label = int(np.argmax(np.bincount(labels)))

    kept_local_idx = np.where(labels == best_label)[0].astype(int)
    kept_pcd = pcd.select_by_index(kept_local_idx)
    return (kept_pcd, kept_local_idx) if return_indices else kept_pcd
