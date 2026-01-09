import pyvista as pv
import numpy as np
from scipy.interpolate import RBFInterpolator
import os
from pathlib import Path


# -----------------------------
# Utility
# -----------------------------

def ensure_normals(mesh):
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(
            inplace=True,
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True
        )


def closest_point_with_normal(surface: pv.PolyData, points):
    """
    Vectorized closest-point query using VTK locator
    """
    locator = pv._vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    ensure_normals(surface)

    hits = []
    normals = []

    for p in points:
        closest = [0.0, 0.0, 0.0]
        cid = pv._vtk.reference(0)
        sid = pv._vtk.reference(0)
        dist2 = pv._vtk.reference(0.0)

        locator.FindClosestPoint(p, closest, cid, sid, dist2)
        hit = np.array(closest)
        hits.append(hit)

        idx = surface.find_closest_point(hit)
        normals.append(surface.point_data["Normals"][idx])

    return np.asarray(hits), np.asarray(normals)


# -----------------------------
# Main algorithm
# -----------------------------

def attach_badge(
    shoe: pv.PolyData,
    badge: pv.PolyData,
    shoe_normal,
    offset=0.05,
    bottom_percent=85,
    max_ctrl=1200,
    rbf_smoothing=0.001,
    n_iters=3,
):
    """
    Robust surface attachment using:
    - closest-point projection
    - hard bottom constraints
    - RBF displacement propagation
    """

    ensure_normals(shoe)
    ensure_normals(badge)

    warped = badge.copy()

    for it in range(n_iters):
        print(f"\n--- Iteration {it+1} ---")

        # -------------------------
        # 1. find bottom region
        # -------------------------
        center = warped.points.mean(axis=0)
        dots = np.dot(warped.points - center, -shoe_normal)
        thr = np.percentile(dots, bottom_percent)
        bottom_idx = np.where(dots > thr)[0]

        print(f"Bottom candidates: {len(bottom_idx)}")

        # -------------------------
        # 2. compute gap
        # -------------------------
        d = warped.compute_implicit_distance(shoe)
        gap = np.abs(d.point_data["implicit_distance"])
        gap_bottom = gap[bottom_idx]

        # pick worst-floating points
        k = min(max_ctrl, len(bottom_idx))
        worst = bottom_idx[np.argsort(-gap_bottom)[:k]]

        ctrl_pts = warped.points[worst]

        # -------------------------
        # 3. closest-point targets
        # -------------------------
        targets, normals = closest_point_with_normal(shoe, ctrl_pts)
        targets = targets + normals * offset

        # -------------------------
        # 4. RBF warp
        # -------------------------
        disp = targets - ctrl_pts

        rbf = RBFInterpolator(
            ctrl_pts,
            disp,
            kernel="linear",
            smoothing=rbf_smoothing
        )

        delta = rbf(warped.points)

        # weight: bottom moves more, top less
        w = np.clip((dots - thr) / (dots.max() - thr + 1e-9), 0, 1)
        warped.points += delta * w[:, None]

        # -------------------------
        # 5. HARD SNAP bottom
        # -------------------------
        warped_d2 = warped.compute_implicit_distance(shoe)
        gap2 = np.abs(warped_d2.point_data["implicit_distance"])

        snap_idx = bottom_idx[gap2[bottom_idx] > offset * 2]

        print(f"Hard snapping {len(snap_idx)} vertices")

        snap_pts = warped.points[snap_idx]
        snap_hits, snap_normals = closest_point_with_normal(shoe, snap_pts)
        warped.points[snap_idx] = snap_hits + snap_normals * offset

        print(
            f"gap max={gap2.max():.3f}, "
            f"p95={np.percentile(gap2,95):.3f}"
        )

    return warped


# -----------------------------
# Entry
# -----------------------------

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    dayong_dir = cur_dir.parent
    shoe_path = os.path.join(dayong_dir, "scans", "STLs", "shoe.stl")
    toy_path = os.path.join(dayong_dir, "scans", "STLs", "badge.stl")

    shoe = pv.read(shoe_path).triangulate().clean()
    badge = pv.read(toy_path).triangulate().clean()

    # scale test
    badge.scale(2.0, inplace=True)

    # choose a representative shoe normal (e.g. toe area)
    ensure_normals(shoe)
    anchor_idx = shoe.n_points // 2
    shoe_normal = shoe.point_data["Normals"][anchor_idx]

    warped_badge = attach_badge(
        shoe,
        badge,
        shoe_normal=shoe_normal,
        offset=0.03,
        bottom_percent=85,
        max_ctrl=1500,
        n_iters=4,
    )

    # visualize
    p = pv.Plotter()
    p.add_mesh(shoe, opacity=0.5, color="lightblue")
    p.add_mesh(warped_badge, color="orange")
    p.show()

    # export
    out = shoe.merge(warped_badge)
    out.save("shoe_with_badge.stl")
    print("Exported shoe_with_badge.stl")