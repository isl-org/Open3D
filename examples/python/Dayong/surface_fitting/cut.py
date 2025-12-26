import numpy as np
import open3d as o3d


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Vector is near-zero")
    return v / n


def split_mesh_by_direction(mesh: o3d.geometry.TriangleMesh,
                            cut_dir: np.ndarray,
                            keep: str = "top",
                            mode: str = "half",
                            band_ratio: float = 0.01):
    """
    Split vertices by a 50% cut along cut_dir.

    We compute s = v·d. s_cut is the midpoint between min and max.

    keep:
      - 'top': keep s > s_cut
      - 'bottom': keep s <= s_cut

    mode:
      - 'half': keep the requested half; slice points are selected by band around s_cut
      - 'band': keep only the band |s-s_cut|<=band

    Returns:
      keep_mask: (N,) bool mask of kept vertices
      slice_idx: indices of vertices near the cut plane (band)
      s_cut: scalar cut threshold
    """
    V = np.asarray(mesh.vertices)
    d = _unit(cut_dir)

    s = V @ d
    s_min, s_max = float(s.min()), float(s.max())
    s_cut = s_min + 0.5 * (s_max - s_min)

    band = band_ratio * (s_max - s_min)
    slice_mask = np.abs(s - s_cut) <= band
    slice_idx = np.where(slice_mask)[0]

    if keep == "top":
        keep_mask = s > s_cut
    elif keep == "bottom":
        keep_mask = s <= s_cut
    else:
        raise ValueError("keep must be 'top' or 'bottom'")

    if mode == "band":
        keep_mask = slice_mask
    elif mode != "half":
        raise ValueError("mode must be 'half' or 'band'")

    return keep_mask, slice_idx, s_cut


def extract_submesh(mesh: o3d.geometry.TriangleMesh, keep_mask: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Extract a submesh containing only triangles whose 3 vertices are in keep_mask.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    keep_mask = np.asarray(keep_mask, dtype=bool)
    tri_keep = keep_mask[F].all(axis=1)
    F_keep = F[tri_keep]

    kept_indices = np.where(keep_mask)[0]
    if len(kept_indices) == 0:
        raise ValueError("No vertices kept; check cut_dir / keep")

    remap = -np.ones(len(V), dtype=np.int32)
    remap[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)

    V_new = V[kept_indices]
    F_new = remap[F_keep]

    sub = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V_new),
        triangles=o3d.utility.Vector3iVector(F_new)
    )
    sub.compute_vertex_normals()
    return sub


def make_debug_pcd(points: np.ndarray, color=(1.0, 0.0, 0.0)) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.paint_uniform_color(list(color))
    return pcd


def cut_preview_mesh(toy: o3d.geometry.TriangleMesh,
                     cut_dir: np.ndarray,
                     keep: str = "top",
                     band_ratio: float = 0.01):
    """Preview the cut result BEFORE projection/warping.

    Returns:
      kept_mesh: submesh containing only the kept half
      dbg: dict with slice_pts (band around cut plane) and metadata

    Notes:
      - This is NOT a geometric boolean cut that creates new cap triangles.
        It simply keeps vertices in the requested half and keeps triangles
        whose 3 vertices all lie in that half.
      - slice_pts are vertices near the cut plane used as the 'driver layer'.
    """
    keep_mask, slice_idx, s_cut = split_mesh_by_direction(
        toy, cut_dir=cut_dir, keep=keep, mode="half", band_ratio=band_ratio
    )
    kept_mesh = extract_submesh(toy, keep_mask)
    dbg = {
        "keep_mask": keep_mask,
        "slice_idx": slice_idx,
        "s_cut": s_cut,
        "slice_pts": np.asarray(toy.vertices)[slice_idx],
    }
    return kept_mesh, dbg


def warp_top_by_projected_slice_closest(
        toy: o3d.geometry.TriangleMesh,
        shoe: o3d.geometry.TriangleMesh,
        cut_dir: np.ndarray,
        move_dir=None,          # ✅ NEW: displacement direction (e.g., -normal_o)
        keep: str = "bottom",
        band_ratio: float = 0.01,
        k: float = 0.1,
        default_t=None,         # ✅ NEW default: clamp extrapolation if None
):
    """
    Minimal-change warp (closest_points on slice band):

    - cut_dir is ONLY used for:
        (a) splitting the toy into top/bottom halves
        (b) defining interpolation coordinate s = v · cut_dir

    - move_dir is used for:
        (a) turning delta = (q - p) into scalar field t = delta · move_dir
        (b) moving vertices: v_new = v + k * t(v) * move_dir

    Returns:
      warped_mesh: TriangleMesh with updated vertices
      dbg: dict for visualization / debugging
    """
    V = np.asarray(toy.vertices)
    F = np.asarray(toy.triangles)

    d_cut = _unit(cut_dir)
    d_move = _unit(cut_dir if move_dir is None else move_dir)

    # --- 1) get slice band indices (driver layer) and keep_mask ---
    keep_mask, slice_idx, s_cut = split_mesh_by_direction(
        toy, cut_dir=d_cut, keep=keep, mode="half", band_ratio=band_ratio
    )
    slice_pts = V[slice_idx]  # (Ns, 3)
    if slice_pts.shape[0] < 3:
        raise ValueError("Slice band too thin / too few points. Increase band_ratio.")

    # --- 2) closest-point projection slice -> shoe ---
    shoe_t = o3d.t.geometry.TriangleMesh.from_legacy(shoe)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(shoe_t)

    q = o3d.core.Tensor(slice_pts.astype(np.float32))  # (Ns, 3)
    ans = scene.compute_closest_points(q)
    projected_slice = ans["points"].numpy()  # (Ns, 3)

    # --- 3) scalar t on slice (project delta onto move_dir) ---
    delta = projected_slice - slice_pts
    t_slice = delta @ d_move  # (Ns,)

    # --- 4) propagate scalar t to kept half by 1D interpolation along s = v·d_cut ---
    s_all = V @ d_cut
    s_slice = s_all[slice_idx]

    # Sort slice samples by s for interpolation
    order = np.argsort(s_slice)
    s_sorted = s_slice[order]
    t_sorted = t_slice[order]

    extent = float(s_sorted.max() - s_sorted.min())
    if extent < 1e-12:
        raise ValueError("Degenerate extent along cut_dir; choose a different cut_dir.")

    # bin duplicates
    bin_size = extent * 1e-4
    key = np.round(s_sorted / max(bin_size, 1e-12)).astype(np.int64)

    uniq_key, inv = np.unique(key, return_inverse=True)
    s_uniq = np.zeros(len(uniq_key), dtype=np.float64)
    t_uniq = np.zeros(len(uniq_key), dtype=np.float64)
    cnt = np.zeros(len(uniq_key), dtype=np.int64)
    for i, g in enumerate(inv):
        s_uniq[g] += s_sorted[i]
        t_uniq[g] += t_sorted[i]
        cnt[g] += 1
    s_uniq /= np.maximum(cnt, 1)
    t_uniq /= np.maximum(cnt, 1)

    # Interpolate only for kept vertices; others stay unchanged
    idx_keep = np.where(keep_mask)[0]
    s_keep = s_all[idx_keep]

    # clamp extrapolation by default
    if default_t is None:
        left_val = float(t_uniq[0])
        right_val = float(t_uniq[-1])
    else:
        left_val = float(default_t)
        right_val = float(default_t)

    t_keep = np.interp(s_keep, s_uniq, t_uniq, left=left_val, right=right_val)

    # --- 5) apply displacement along move_dir ---
    V_new = V.copy()
    V_new[idx_keep] = V_new[idx_keep] + (k * t_keep)[:, None] * d_move[None, :]

    warped = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V_new),
        triangles=o3d.utility.Vector3iVector(F)
    )
    warped.compute_vertex_normals()

    dbg = {
        "keep_mask": keep_mask,
        "slice_idx": slice_idx,
        "s_cut": s_cut,
        "slice_pts": slice_pts,
        "projected_slice": projected_slice,
        "delta": delta,
        "t_slice": t_slice,
        "cut_dir_unit": d_cut,
        "move_dir_unit": d_move,
    }
    return warped, dbg