import numpy as np
import open3d as o3d

from typing import Tuple, Optional

from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from matplotlib.path import Path


# ============================================================
# 1) Boundary loop utilities (concave/alpha-shape, ordered loop)
# ============================================================

def _alpha_shape_boundary_edges(points2d: np.ndarray, alpha: float):
    """Return boundary edges (i,j) for the alpha shape of 2D points."""
    tri = Delaunay(points2d)

    edges = {}

    def _circumradius(pa, pb, pc):
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area2 = s * (s - a) * (s - b) * (s - c)
        if area2 <= 1e-12:
            return np.inf
        area = np.sqrt(area2)
        return (a * b * c) / (4.0 * area)

    for simplex in tri.simplices:
        ia, ib, ic = simplex
        pa, pb, pc = points2d[ia], points2d[ib], points2d[ic]
        cr = _circumradius(pa, pb, pc)

        if cr < alpha:
            for (u, v) in [(ia, ib), (ib, ic), (ic, ia)]:
                key = tuple(sorted((int(u), int(v))))
                edges[key] = edges.get(key, 0) + 1

    # boundary edges appear only once
    boundary_edges = [(u, v) for (u, v), cnt in edges.items() if cnt == 1]
    return boundary_edges


def _order_edges_into_loop(n_points: int, boundary_edges: list[tuple[int, int]]) -> np.ndarray:
    """Order boundary edges into a single loop of vertex indices.

    Assumes the boundary is mostly one closed loop.
    """
    if len(boundary_edges) == 0:
        return np.array([], dtype=int)

    # adjacency
    adj: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # pick a stable start: the left-most point among boundary vertices
    boundary_verts = np.array(sorted(adj.keys()), dtype=int)

    # if something went wrong and we have too few vertices
    if len(boundary_verts) < 3:
        return boundary_verts

    # build loop by walking neighbor-to-neighbor
    # choose start as the smallest x (we don't have points here; caller will reorder by coords if needed)
    start = int(boundary_verts[0])

    loop = [start]
    prev = None
    cur = start

    # walk until we return to start or get stuck
    for _ in range(len(boundary_verts) + 5):
        nbrs = adj.get(cur, [])
        if len(nbrs) == 0:
            break

        # choose next not equal prev if possible
        if prev is None:
            nxt = nbrs[0]
        else:
            cand = [n for n in nbrs if n != prev]
            nxt = cand[0] if len(cand) > 0 else nbrs[0]

        if nxt == start:
            break

        loop.append(int(nxt))
        prev, cur = cur, int(nxt)

    return np.array(loop, dtype=int)


def compute_ordered_boundary_loop_xy(
    boundary_xy: np.ndarray,
    alpha: float = 12.0,
    simplify_every: int = 1,
) -> np.ndarray:
    """Given (unordered) boundary points in XY, compute an ORDERED boundary loop.

    Why this exists:
      - Using ConvexHull will often pick the wrong arc because the insole outline is NOT convex.
      - We need an ordered, concave-aware outer boundary to splice the "missing wall" reliably.

    Parameters
    ----------
    boundary_xy : (N,2)
        Points on (or near) the *outer* boundary in XY.
    alpha : float
        Alpha radius (in your XY units, mm) controlling concavity.
        Bigger alpha -> closer to convex hull; smaller alpha -> more concave, but can fragment.
    simplify_every : int
        Optional downsample of the loop after ordering (keep every k-th point).

    Returns
    -------
    loop_xy : (M,2)
        Ordered boundary loop (not repeated first point at the end).
    """
    boundary_xy = np.asarray(boundary_xy, dtype=float)
    if len(boundary_xy) < 10:
        return boundary_xy

    edges = _alpha_shape_boundary_edges(boundary_xy, alpha=alpha)
    if len(edges) == 0:
        # fallback: nearest-neighbor ordering
        pts = boundary_xy
        used = np.zeros(len(pts), dtype=bool)
        start = int(np.argmin(pts[:, 0]))
        order = [start]
        used[start] = True
        for _ in range(len(pts) - 1):
            cur = order[-1]
            d = np.linalg.norm(pts - pts[cur], axis=1)
            d[used] = np.inf
            nxt = int(np.argmin(d))
            order.append(nxt)
            used[nxt] = True
        loop = pts[np.array(order, dtype=int)]
        return loop[:: max(1, simplify_every)]

    loop_idx = _order_edges_into_loop(len(boundary_xy), edges)
    loop_xy = boundary_xy[loop_idx]

    # rotate loop so it starts at left-most x (more stable for slicing)
    start = int(np.argmin(loop_xy[:, 0]))
    loop_xy = np.roll(loop_xy, -start, axis=0)

    if simplify_every > 1:
        loop_xy = loop_xy[::simplify_every]

    return loop_xy


# ============================================================
# 2) Fit + extend the pre-cut line to the boundary
# ============================================================

def fit_and_extend_pre_cut_line(
    pre_cut_line_pcd: o3d.geometry.PointCloud,
    boundary_loop_xy: np.ndarray,
    extend_margin: float = 6.0,
    n_samples: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a smooth 2D curve to the pre-cut line points (XY),
    then extend it in both directions until it reaches the boundary.

    NOTE:
      We DO NOT use convex hull here. boundary_loop_xy must be an ordered outer loop.

    Returns
    -------
    extended_curve_xy : (K,2)
        curve with two endpoints snapped to boundary
    end_pts_xy : (2,2)
        endpoints on the boundary
    """
    pts = np.asarray(pre_cut_line_pcd.points)
    if len(pts) < 10:
        raise ValueError("pre_cut_line_pcd has too few points to fit")

    xy = pts[:, :2]

    # PCA provides a stable 1D parameter t along the curve
    pca = PCA(n_components=2)
    uv = pca.fit_transform(xy)

    t = uv[:, 0]
    x = xy[:, 0]
    y = xy[:, 1]

    # Sort by t for stable interpolation
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]

    # Fit quadratic y(t)
    coeffs = np.polyfit(t, y, deg=2)
    poly = np.poly1d(coeffs)

    t_min, t_max = float(t.min()), float(t.max())
    t_ext = np.linspace(t_min - extend_margin, t_max + extend_margin, n_samples)

    y_ext = poly(t_ext)
    # x(t) via 1D interpolation
    x_ext = np.interp(t_ext, t, x)

    curve_xy = np.column_stack([x_ext, y_ext])

    # Snap the two ends to the nearest points on the boundary loop
    d0 = np.linalg.norm(boundary_loop_xy - curve_xy[0], axis=1)
    d1 = np.linalg.norm(boundary_loop_xy - curve_xy[-1], axis=1)

    p0 = boundary_loop_xy[int(np.argmin(d0))]
    p1 = boundary_loop_xy[int(np.argmin(d1))]

    extended_curve_xy = np.vstack([p0, curve_xy, p1])
    end_pts_xy = np.vstack([p0, p1])

    return extended_curve_xy, end_pts_xy


# ============================================================
# 3) Build two candidate polygons, choose smaller XY area
# ============================================================

def _polygon_area_xy(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def build_precut_polygon_smaller_area(
    extended_curve_xy: np.ndarray,
    boundary_loop_xy: np.ndarray,
) -> np.ndarray:
    """Combine the extended curve + boundary loop to form TWO possible closed polygons.

    We slice the ordered boundary loop between the two snapped endpoints in two ways (forward/backward),
    then attach the reversed curve to close the loop.

    Returns the polygon with the smaller XY area.
    """
    if len(boundary_loop_xy) < 10:
        raise ValueError("boundary_loop_xy too small")

    p0 = extended_curve_xy[0]
    p1 = extended_curve_xy[-1]

    # Find closest boundary indices to endpoints
    i0 = int(np.argmin(np.linalg.norm(boundary_loop_xy - p0, axis=1)))
    i1 = int(np.argmin(np.linalg.norm(boundary_loop_xy - p1, axis=1)))

    m = len(boundary_loop_xy)

    # forward path along boundary from i0 -> i1
    if i0 <= i1:
        arc_fwd = boundary_loop_xy[i0:i1 + 1]
        arc_bwd = np.vstack([boundary_loop_xy[i1:], boundary_loop_xy[:i0 + 1]])
    else:
        arc_fwd = np.vstack([boundary_loop_xy[i0:], boundary_loop_xy[:i1 + 1]])
        arc_bwd = boundary_loop_xy[i1:i0 + 1]

    curve_rev = extended_curve_xy[::-1]

    poly_fwd = np.vstack([arc_fwd, curve_rev])
    poly_bwd = np.vstack([arc_bwd, curve_rev])

    area_fwd = _polygon_area_xy(poly_fwd)
    area_bwd = _polygon_area_xy(poly_bwd)

    # choose smaller
    poly = poly_fwd if area_fwd <= area_bwd else poly_bwd

    # (optional) remove consecutive duplicates (helps Path.contains_points stability)
    diff = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    keep = np.ones(len(poly), dtype=bool)
    keep[1:] = diff > 1e-9
    poly = poly[keep]

    return poly


# ============================================================
# 4) Extract pre-cut region points (point-in-polygon)
# ============================================================

def extract_precut_region_from_polygon(
    top_pcd: o3d.geometry.PointCloud,
    polygon_xy: np.ndarray,
    visualize: bool = True,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Given a closed polygon in XY, return all points of top_pcd inside it."""
    pts = np.asarray(top_pcd.points)
    pts_xy = pts[:, :2]

    path = Path(polygon_xy)
    inside_mask = path.contains_points(pts_xy)

    region_idx = np.where(inside_mask)[0]
    region_pcd = top_pcd.select_by_index(region_idx)

    if visualize:
        colors = np.full((len(pts), 3), 0.7)
        colors[region_idx] = np.array([1.0, 0.0, 0.0])

        vis = o3d.geometry.PointCloud(top_pcd)
        vis.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries(
            [vis],
            window_name="Pre-cut region (red)",
            width=1400,
            height=900,
            mesh_show_back_face=True,
        )

    return region_pcd, region_idx


# ============================================================
# 5) Debug visualization (curve + polygon)
# ============================================================

def visualize_precut_geometry(
    top_pcd: o3d.geometry.PointCloud,
    curve_xy: np.ndarray,
    polygon_xy: np.ndarray,
):
    pts = np.asarray(top_pcd.points)
    z0 = float(np.mean(pts[:, 2])) if len(pts) > 0 else 0.0

    vis = o3d.geometry.PointCloud(top_pcd)
    vis.paint_uniform_color([0.6, 0.6, 0.6])

    curve_3d = np.column_stack([curve_xy, np.full(len(curve_xy), z0)])
    poly_3d = np.column_stack([polygon_xy, np.full(len(polygon_xy), z0)])

    curve_pcd = o3d.geometry.PointCloud()
    curve_pcd.points = o3d.utility.Vector3dVector(curve_3d)
    curve_pcd.paint_uniform_color([0.0, 0.0, 0.0])  # black

    poly_pcd = o3d.geometry.PointCloud()
    poly_pcd.points = o3d.utility.Vector3dVector(poly_3d)
    poly_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # green

    o3d.visualization.draw_geometries(
        [vis, curve_pcd, poly_pcd],
        window_name="Pre-cut curve + chosen polygon",
        mesh_show_back_face=True,
    )


# ============================================================
# 6) One-call helper for unit code
# ============================================================

def extract_precut_region(
    top_pcd: o3d.geometry.PointCloud,
    pre_cut_line_pcd: o3d.geometry.PointCloud,
    outer_boundary_xy: np.ndarray,
    alpha: float = 12.0,
    extend_margin: float = 6.0,
    visualize: bool = True,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    """High-level pipeline:

    1) Order the outer boundary loop (concave-aware)
    2) Fit + extend pre-cut curve to boundary
    3) Build TWO candidate polygons, choose smaller area
    4) Extract region points inside polygon

    Returns
    -------
    region_pcd, region_idx, extended_curve_xy, polygon_xy
    """
    boundary_loop_xy = compute_ordered_boundary_loop_xy(outer_boundary_xy, alpha=alpha)

    extended_curve_xy, _ = fit_and_extend_pre_cut_line(
        pre_cut_line_pcd,
        boundary_loop_xy=boundary_loop_xy,
        extend_margin=extend_margin,
    )

    polygon_xy = build_precut_polygon_smaller_area(
        extended_curve_xy=extended_curve_xy,
        boundary_loop_xy=boundary_loop_xy,
    )

    if visualize:
        visualize_precut_geometry(top_pcd, extended_curve_xy, polygon_xy)

    region_pcd, region_idx = extract_precut_region_from_polygon(
        top_pcd,
        polygon_xy=polygon_xy,
        visualize=visualize,
    )

    return region_pcd, region_idx, extended_curve_xy, polygon_xy