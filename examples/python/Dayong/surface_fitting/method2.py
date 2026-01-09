import pyvista as pv
import numpy as np
from scipy.interpolate import RBFInterpolator
import os
from pathlib import Path

# Pick an anchor point on a mesh using bbox ratios
def get_point_by_ratio(mesh, x_ratio, y_ratio, z_ratio):
    b = mesh.bounds
    target = np.array([
        b[0] + (b[1] - b[0]) * x_ratio,
        b[2] + (b[3] - b[2]) * y_ratio,
        b[4] + (b[5] - b[4]) * z_ratio
    ])
    idx = mesh.find_closest_point(target)
    return mesh.points[idx]

# Get local normal (via nearest vertex normal)
def get_normal_at_point(mesh, point):
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(inplace=True, point_normals=True, cell_normals=False)
    idx = mesh.find_closest_point(point)
    normal = mesh.point_data["Normals"][idx]
    # Normalize to unit vector for stable geometric computations
    return normal / np.linalg.norm(normal)

# Rigid alignment (rotate + translate) toy onto shoe
def align_toy_to_shoe(shoe, toy, o_ratio, p_ratio):
    o = get_point_by_ratio(shoe, *o_ratio)
    normal_o = get_normal_at_point(shoe, o)
    p = get_point_by_ratio(toy, *p_ratio)
    normal_p = get_normal_at_point(toy, p)

    # Calculate Rotation

    # Goal: rotate toy so that its "attachment side" faces the shoe surface normal
    v_from = -normal_p
    v_to = normal_o

    axis = np.cross(v_from, v_to)
    axis_len = np.linalg.norm(axis)

    if axis_len > 1e-6:
        axis = axis / axis_len
        angle = np.degrees(np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0)))
        toy.rotate_vector(vector=axis, angle=angle, point=p, inplace=True)

    # Translate P to O
    toy.translate(o - p + normal_o * 5, inplace=True)
    return o, normal_o

# Surface fitting using RBF displacement field
def apply_non_rigid_warp(shoe, toy, shoe_normal, show_steps=True, offset=0.5,
                         n_iters=4, bottom_percent=90, max_samples=500,
                         smoothing=0.01):
    """
        Core idea:
        1) Identify candidate points on the "bottom/attachment" side of the toy.
        2) Measure which of those points are most floating above the shoe (largest gap).
        3) For those points, ray-cast to shoe to get target contact points.
        4) Fit a smooth displacement field using RBF so the entire toy deforms smoothly.
        5) Iterate to progressively reduce remaining gaps on concave surfaces.
    """

    # 0) Resolution: increase degrees of freedom so toy can bend to match shoe curvature
    if toy.n_points < 5000:
        toy = toy.subdivide(1)

    # Ensure shoe normals exist (for local offset direction at hit points)
    if "Normals" not in shoe.point_data:
        shoe.compute_normals(inplace=True, point_normals=True, cell_normals=False)

    warped = toy.copy()

    for it in range(n_iters):
        # 1) Select candidate attachment points on toy (bottom side)
        toy_center = np.mean(warped.points, axis=0)

        # dots = projection of each vertex onto the direction pointing INTO the shoe (-shoe_normal)
        # larger dots => vertex is more on the "bottom/attachment" side
        dots = np.dot(warped.points - toy_center, -shoe_normal)

        # bottom_percent controls how much of the attachment side we consider:
        # e.g., 90 => bottom 10%; 70 => bottom 30% (stronger constraints, but may affect more geometry)
        cand_idx = np.where(dots > np.percentile(dots, bottom_percent))[0]  # bottom (100-bottom_percent)%

        if cand_idx.size == 0:
            print(f"[iter {it}] No candidates found.")
            break

        # 2) Compute gap between toy candidate points and the shoe surface
        # implicit_distance is signed distance to the shoe surface; abs() gives absolute gap value
        warped_d = warped.compute_implicit_distance(shoe)
        dist_all = np.abs(warped_d.point_data["implicit_distance"])  # unsigned gap
        cand_dist = dist_all[cand_idx]

        # 3) Prioritize the MOST floating points (largest gap) as control points
        k = min(max_samples, cand_idx.size)
        far_order = np.argsort(-cand_dist)[:k]  # descending by gap
        base_idx = cand_idx[far_order]
        base_pts = warped.points[base_idx]

        # 4) Ray-cast each control point to shoe to get target
        targets = []
        valid_idx = []

        for i, pt in enumerate(base_pts):
            # Ray along shoe_normal direction: start outside, shoot inward
            start = pt + shoe_normal * 10
            stop  = pt - shoe_normal * 30
            hits, _ = shoe.ray_trace(start, stop)

            if len(hits) > 0:
                hit = hits[0] # first intersection with shoe

                # Use local normal at hit point for offset; important on curved/concave surfaces
                n_local = get_normal_at_point(shoe, hit)
                targets.append(hit + n_local * offset)
                valid_idx.append(base_idx[i])

        if len(targets) < 10:
            print(f"[iter {it}] Too few hits ({len(targets)}). Stop.")
            break

        # control_pts are on toy; target_pts are on/near shoe surface
        control_pts = warped.points[valid_idx]
        target_pts  = np.asarray(targets)

        # displacement vectors we want to enforce at control points
        disp = target_pts - control_pts

        # 5) Fit a smooth displacement field with RBF and warp all vertices
        # RBFInterpolator learns f(x) = displacement at position x.
        interpolator = RBFInterpolator(
            control_pts,
            disp,
            kernel="linear",
            smoothing=smoothing
        )

        # Apply the deformation field to all vertices (smooth non-rigid warp)
        warped.points += interpolator(warped.points)

        # Debug visualize only first iter (optional)
        # if show_steps and it == 0:
        #     p = pv.Plotter()
        #     p.add_text("Controls (red) are FAR gaps; Targets (green) on shoe", font_size=10)
        #     p.add_mesh(shoe, color="white", opacity=0.3)
        #     p.add_mesh(warped, color="orange", opacity=0.7)
        #     p.add_points(control_pts, color="red", point_size=4)
        #     p.add_points(target_pts, color="green", point_size=4)
            # p.show()

        # Optional: quick progress print
        print(f"[iter {it}] candidates={cand_idx.size}, controls={len(valid_idx)}, max_gap={cand_dist.max():.4f}")

    return warped

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    dayong_dir = cur_dir.parent
    shoe_path = os.path.join(dayong_dir, "scans", "STLs", "shoe.stl")
    toy_path = os.path.join(dayong_dir, "scans", "STLs", "badge.stl")

    if not os.path.exists(shoe_path) or not os.path.exists(toy_path):
        print("Error: Files not found.")
    else:
        # 1. Load
        shoe_mesh = pv.read(shoe_path)
        toy_mesh = pv.read(toy_path)

        # Scale toy for better visualization
        # toy_mesh.scale(2, inplace=True)

        # p1 = pv.Plotter()
        # p1.add_text("Step 1: Initial State (Before Alignment)", font_size=10)
        # p1.add_mesh(shoe_mesh, color="lightblue", opacity=0.5)
        # p1.add_mesh(toy_mesh, color="orange")
        # p1.show()

        # 2. Align (rotate & translate)
        o_pt, n_o = align_toy_to_shoe(shoe_mesh, toy_mesh, (0.5, 0.2, 0.9), (0.5, 0.5, 0.0)) # front

        # p2 = pv.Plotter()
        # p2.add_text("Step 2: Post-Alignment (Positioned at O)", font_size=10)
        # p2.add_mesh(shoe_mesh, color="lightblue", opacity=0.5)
        # p2.add_mesh(toy_mesh, color="orange")
        # p2.add_arrows(o_pt, n_o, mag=10, color="red")
        # p2.show()

        # 3. Warp
        warped_toy = apply_non_rigid_warp(
            shoe_mesh, toy_mesh, n_o,
            show_steps=True,
            offset=0.5,
            n_iters=8,
            bottom_percent=70,
            max_samples=600,
            smoothing=0.01
        )

        # 4. Final Visualization
        p4 = pv.Plotter()
        p4.add_text("Step 4: Final Non-Rigid Warp Result", font_size=10)
        p4.add_mesh(shoe_mesh, color="lightblue", opacity=0.8)
        p4.add_mesh(warped_toy, color="orange")
        p4.show()

        # 5. Export to STL
        # warped_toy.compute_normals(inplace=True, flip_normals=False)
        combined = shoe_mesh.merge(warped_toy)
        combined.save("warp.stl")
        print(f"STL Exported Successfully!")