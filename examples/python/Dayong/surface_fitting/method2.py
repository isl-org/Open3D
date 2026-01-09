import os
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import RBFInterpolator


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def get_point_by_ratio(mesh: pv.PolyData, x_ratio: float, y_ratio: float, z_ratio: float) -> np.ndarray:
    """Pick a point on a mesh by bbox ratios (0~1 in x/y/z)."""
    b = mesh.bounds
    target = np.array([
        b[0] + (b[1] - b[0]) * x_ratio,
        b[2] + (b[3] - b[2]) * y_ratio,
        b[4] + (b[5] - b[4]) * z_ratio,
    ])
    idx = mesh.find_closest_point(target)
    return mesh.points[idx]


def ensure_point_normals(mesh: pv.PolyData) -> None:
    """Ensure mesh has point normals in mesh.point_data['Normals']."""
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(inplace=True, point_normals=True, cell_normals=False, auto_orient_normals=True)


def get_normal_at_point(mesh: pv.PolyData, point: np.ndarray) -> np.ndarray:
    """Approximate local normal by nearest vertex normal."""
    ensure_point_normals(mesh)
    idx = mesh.find_closest_point(point)
    n = mesh.point_data["Normals"][idx]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n


def build_closest_point_locator(surface: pv.PolyData):
    """VTK locator for fast closest-point queries."""
    locator = pv._vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    return locator


def closest_points_and_normals(surface: pv.PolyData, locator, query_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query point, return closest point on surface + its local normal (nearest vertex normal)."""
    ensure_point_normals(surface)

    out_pts = np.empty_like(query_pts)
    out_ns = np.empty_like(query_pts)

    for i, p in enumerate(query_pts):
        closest = [0.0, 0.0, 0.0]
        cid = pv._vtk.reference(0)
        sid = pv._vtk.reference(0)
        dist2 = pv._vtk.reference(0.0)
        locator.FindClosestPoint(p, closest, cid, sid, dist2)
        hit = np.array(closest, dtype=float)

        out_pts[i] = hit
        out_ns[i] = get_normal_at_point(surface, hit)

    # normalize normals
    out_ns /= (np.linalg.norm(out_ns, axis=1, keepdims=True) + 1e-12)
    return out_pts, out_ns


# ------------------------------------------------------------
# Rigid alignment (rotate + translate) toy onto shoe
# ------------------------------------------------------------

def align_toy_to_shoe(shoe: pv.PolyData, toy: pv.PolyData, o_ratio, p_ratio):
    """Rotate toy so its anchor normal faces the shoe anchor normal, then translate anchor-to-anchor."""
    o = get_point_by_ratio(shoe, *o_ratio)
    normal_o = get_normal_at_point(shoe, o)

    p = get_point_by_ratio(toy, *p_ratio)
    normal_p = get_normal_at_point(toy, p)

    v_from = -normal_p
    v_to = normal_o

    axis = np.cross(v_from, v_to)
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-8:
        axis /= axis_len
        angle = np.degrees(np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0)))
        toy.rotate_vector(vector=axis, angle=angle, point=p, inplace=True)

    # small lift above surface for stability before warp
    toy.translate(o - p + normal_o * 5.0, inplace=True)
    return o, normal_o


# ------------------------------------------------------------
# Non-rigid attachment (Bottom conforms, Top preserves thickness/details)
# ------------------------------------------------------------

def apply_non_rigid_warp(
    shoe: pv.PolyData,
    toy: pv.PolyData,
    shoe_normal: np.ndarray,
    *,
    offset: float = 0.2,
    n_iters: int = 8,
    max_ctrl: int = 1200,
    bottom_frac: float = 0.35,
    rbf_kernel: str = "linear",
    rbf_smoothing: float = 0.002,
    hard_snap_gap: float = 0.15,
    preserve_alpha: float = 0.9,
    show_debug: bool = False,
) -> pv.PolyData:
    """Attach toy to shoe surface robustly.

    Key idea:
    - Identify the "bottom" region of the toy along -shoe_normal.
    - Choose control points among bottom vertices with the largest gap to the shoe.
    - Compute targets using *closest point* on shoe (direction-free; works on concave areas).
    - Use RBF to propagate the displacement field to bottom + transition.
    - Preserve details/thickness by NOT freezing the top in world space.
      Instead, store each TOP vertex's offset relative to its nearest BOTTOM vertex,
      then re-apply: top_new = bottom_new(nearest) + offset_saved.

    This eliminates the "extruded walls/spikes" artifact.
    """

    # Clean input (triangulate makes distance/locators more stable)
    shoe = shoe.triangulate().clean()
    toy = toy.triangulate().clean()
    ensure_point_normals(shoe)
    ensure_point_normals(toy)

    # Safety: avoid insane vertex counts (your badge can be ~1M points after subdivision)
    # Keep enough geometry for details, but prevent OOM / super slow operations.
    max_points = 220_000
    if toy.n_points > max_points:
        target_reduction = 1.0 - (max_points / float(toy.n_points))
        target_reduction = np.clip(target_reduction, 0.0, 0.97)
        print(f"[prep] Toy has {toy.n_points} points -> decimate_pro(reduction={target_reduction:.3f})")
        toy = toy.decimate_pro(reduction=target_reduction, preserve_topology=True)
        toy = toy.clean().triangulate()
        ensure_point_normals(toy)

    warped = toy.copy()

    # Build one locator for the shoe (fast closest-point queries)
    locator = build_closest_point_locator(shoe)

    # --- Split toy into bottom/top using projection along -shoe_normal ---
    center0 = warped.points.mean(axis=0)
    proj0 = np.dot(warped.points - center0, -shoe_normal)

    # bottom_frac=0.35 means: take bottom 35% (largest projection values)
    proj_thr = np.percentile(proj0, 100.0 * (1.0 - bottom_frac))
    bottom_indices = np.where(proj0 >= proj_thr)[0]
    top_indices = np.where(proj0 < proj_thr)[0]

    print(f"[split] bottom={len(bottom_indices)} ({100*len(bottom_indices)/warped.n_points:.1f}%), "
          f"top={len(top_indices)} ({100*len(top_indices)/warped.n_points:.1f}%)")

    # --- Preserve thickness/details: map each TOP vertex -> nearest BOTTOM vertex and store relative offset ---
    # NOTE: freezing TOP in world space while bottom moves creates tall side-walls (the artifact you saw).
    bottom_pts0 = warped.points[bottom_indices]
    top_pts0 = warped.points[top_indices]

    # Some PyVista/VTK builds don't expose vtkKdTreePointLocator; vtkPointLocator is widely available.
    kdt = pv._vtk.vtkPointLocator()
    bottom_pd = pv.PolyData(bottom_pts0)
    kdt.SetDataSet(bottom_pd)
    kdt.BuildLocator()

    top_to_bottom_local = np.empty(len(top_indices), dtype=np.int64)
    for ii, tp in enumerate(top_pts0):
        top_to_bottom_local[ii] = int(kdt.FindClosestPoint(tp))

    top_rel_offsets = top_pts0 - bottom_pts0[top_to_bottom_local]

    # --- iterative attachment ---
    bbox = np.array(warped.bounds)
    bbox_diag = float(np.linalg.norm(bbox[1::2] - bbox[0::2]))

    for it in range(n_iters):
        # recompute projections each iter (geometry changes)
        center = warped.points.mean(axis=0)
        proj = np.dot(warped.points - center, -shoe_normal)

        # implicit distance gives SIGNED distance; positive often means outside depending on mesh orientation.
        # We use absolute gap magnitude for ranking.
        d = warped.compute_implicit_distance(shoe)
        gap = np.abs(d.point_data["implicit_distance"])

        # Choose controls from bottom that are farthest from shoe (largest gap)
        bottom_gap = gap[bottom_indices]
        k = min(max_ctrl, len(bottom_indices))
        far_local = np.argsort(-bottom_gap)[:k]
        ctrl_idx = bottom_indices[far_local]

        ctrl_pts = warped.points[ctrl_idx]

        # Target = closest point on shoe + local normal * offset
        hit_pts, hit_ns = closest_points_and_normals(shoe, locator, ctrl_pts)
        targets = hit_pts + hit_ns * offset

        disp = targets - ctrl_pts

        # RBF displacement field
        rbf = RBFInterpolator(
            ctrl_pts,
            disp,
            kernel=rbf_kernel,
            smoothing=rbf_smoothing,
        )

        # Apply to bottom + a small transition band above bottom
        proj_range = float(proj.max() - proj.min() + 1e-12)
        trans_thr = proj_thr - 0.10 * proj_range
        transition_indices = np.where((proj >= trans_thr) & (proj < proj_thr))[0]

        # Bottom gets full displacement
        bottom_disp = rbf(warped.points[bottom_indices])

        # Safety clamp to avoid rare RBF explosions
        max_step = bbox_diag * 0.10
        mags = np.linalg.norm(bottom_disp, axis=1)
        too_big = mags > max_step
        if np.any(too_big):
            bottom_disp[too_big] *= (max_step / (mags[too_big] + 1e-12))[:, None]

        warped.points[bottom_indices] += bottom_disp

        # Transition gets blended displacement (smooth connection)
        if len(transition_indices) > 0:
            w = (proj[transition_indices] - trans_thr) / (proj_thr - trans_thr + 1e-12)
            w = np.clip(w, 0.0, 1.0)
            trans_disp = rbf(warped.points[transition_indices])
            warped.points[transition_indices] += trans_disp * w[:, None]

        # HARD SNAP (optional): if some bottom points are still floating, snap them directly to closest point
        d2 = warped.compute_implicit_distance(shoe)
        gap2 = np.abs(d2.point_data["implicit_distance"])
        snap_mask = gap2[bottom_indices] > hard_snap_gap
        snap_idx = bottom_indices[np.where(snap_mask)[0]]
        if len(snap_idx) > 0:
            snap_pts = warped.points[snap_idx]
            snap_hits, snap_ns = closest_points_and_normals(shoe, locator, snap_pts)
            warped.points[snap_idx] = snap_hits + snap_ns * offset

        # Re-apply thickness/detail preservation for TOP
        bottom_pts_now = warped.points[bottom_indices]
        top_target = bottom_pts_now[top_to_bottom_local] + top_rel_offsets
        warped.points[top_indices] = (1.0 - preserve_alpha) * warped.points[top_indices] + preserve_alpha * top_target

        # Logging
        p95 = float(np.percentile(gap2, 95))
        print(
            f"[iter {it:02d}] ctrl={len(ctrl_idx)} | gap: mean={gap2.mean():.3f} p95={p95:.3f} max={gap2.max():.3f} | hard_snap={len(snap_idx)}"
        )

        if show_debug and it == 0:
            pl = pv.Plotter()
            pl.add_text("Debug: control points (red) and targets (green)")
            pl.add_mesh(shoe, color="lightblue", opacity=0.35)
            pl.add_mesh(warped, color="orange", opacity=0.8)
            pl.add_points(ctrl_pts, color="red", point_size=4)
            pl.add_points(targets, color="green", point_size=4)
            pl.show()

    return warped


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    dayong_dir = cur_dir.parent
    shoe_path = os.path.join(dayong_dir, "scans", "STLs", "shoe.stl")
    toy_path = os.path.join(dayong_dir, "scans", "STLs", "badge.stl")

    if not os.path.exists(shoe_path) or not os.path.exists(toy_path):
        print("Error: Files not found.")
        print(f"Looking for:\n  {shoe_path}\n  {toy_path}")
        raise SystemExit(1)

    shoe_mesh = pv.read(shoe_path)
    toy_mesh = pv.read(toy_path)

    # ---- Test scaling ----
    print("=" * 60)
    print("TESTING WITH 2X SCALED BADGE")
    print("=" * 60)
    toy_mesh.scale(2.0, inplace=True)

    # ---- Rigid pre-alignment ----
    o_pt, n_o = align_toy_to_shoe(
        shoe_mesh,
        toy_mesh,
        (0.5, 0.2, 0.9),  # Shoe anchor
        (0.5, 0.5, 0.0),  # Badge anchor
    )

    # ---- Warp attach ----
    warped_toy = apply_non_rigid_warp(
        shoe_mesh,
        toy_mesh,
        n_o,
        offset=0.15,
        n_iters=8,
        max_ctrl=1400,
        bottom_frac=0.40,
        rbf_kernel="linear",
        rbf_smoothing=0.003,
        hard_snap_gap=0.12,
        preserve_alpha=0.9,
        show_debug=False,
    )

    # ---- Visualize ----
    p = pv.Plotter()
    p.add_text("Final Result: Badge attached to shoe")
    p.add_mesh(shoe_mesh, color="lightblue", opacity=0.8)
    p.add_mesh(warped_toy, color="orange", show_edges=False)
    p.show()

    # ---- Quality metrics ----
    final_dist = warped_toy.compute_implicit_distance(shoe_mesh)
    gaps = np.abs(final_dist.point_data["implicit_distance"])
    print("\n" + "=" * 60)
    print("QUALITY METRICS")
    print(f"  mean gap: {gaps.mean():.4f} mm")
    print(f"  p95  gap: {np.percentile(gaps, 95):.4f} mm")
    print(f"  max  gap: {gaps.max():.4f} mm")
    print("=" * 60)

    # ---- Export ----
    combined = shoe_mesh.merge(warped_toy)
    out_path = os.path.join(cur_dir, "attached_badge_2x.stl")
    combined.save(out_path)
    print(f"Exported: {out_path}")