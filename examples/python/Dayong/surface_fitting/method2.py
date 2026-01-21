import pyvista as pv
import numpy as np
from scipy.interpolate import RBFInterpolator
import os
from pathlib import Path


# -----------------------------------------------------------------------------
# Reporting & Helpers
# -----------------------------------------------------------------------------

def print_mesh_stats(mesh, name="Mesh"):
    """Prints dimensions and center of a mesh."""
    b = mesh.bounds
    # Dimensions: Length (X), Width (Y), Height (Z)
    dims = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    center = mesh.center
    diag = np.linalg.norm(dims)

    print(f"--- {name} Stats ---")
    print(f"  Points:     {mesh.n_points}")
    print(f"  Bounds:     X[{b[0]:.2f}, {b[1]:.2f}] Y[{b[2]:.2f}, {b[3]:.2f}] Z[{b[4]:.2f}, {b[5]:.2f}]")
    print(f"  Dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} (Diagonal: {diag:.2f})")
    print(f"  Center:     [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print("")


def get_point_by_ratio(mesh, x_ratio, y_ratio, z_ratio):
    b = mesh.bounds
    target = np.array([
        b[0] + (b[1] - b[0]) * x_ratio,
        b[2] + (b[3] - b[2]) * y_ratio,
        b[4] + (b[5] - b[4]) * z_ratio
    ])
    idx = mesh.find_closest_point(target)
    return mesh.points[idx]


def get_normal_at_point(mesh, point):
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(inplace=True, point_normals=True, cell_normals=False)
    idx = mesh.find_closest_point(point)
    normal = mesh.point_data["Normals"][idx]
    return normal / np.linalg.norm(normal)


def project_points_along_vector(target_mesh, origins, direction):
    """
    Robustly project points 'origins' along vector 'direction' to hit 'target_mesh'.
    Includes Debug prints for Normal Flipping.
    """
    locator = pv._vtk.vtkOBBTree()
    locator.SetDataSet(target_mesh)
    locator.BuildLocator()

    hits = np.zeros_like(origins)
    hit_normals = np.zeros_like(origins)
    valid_mask = np.zeros(len(origins), dtype=bool)

    if "Normals" not in target_mesh.cell_data:
        target_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
    target_cell_normals = target_mesh.cell_data["Normals"]

    # Calculate safe ray length based on target size
    diag = np.linalg.norm(np.array(target_mesh.bounds[1::2]) - np.array(target_mesh.bounds[0::2]))
    ray_len = diag * 1.5
    step = direction * ray_len

    points_vtk = pv._vtk.vtkPoints()
    cell_ids_vtk = pv._vtk.vtkIdList()

    flip_count = 0

    for i, start_pt in enumerate(origins):
        end_pt = start_pt + step
        if locator.IntersectWithLine(start_pt, end_pt, points_vtk, cell_ids_vtk) > 0:
            p = np.array(points_vtk.GetPoint(0))
            cell_id = cell_ids_vtk.GetId(0)
            n = np.array(target_cell_normals[cell_id])

            # Normal Flip Logic
            if np.dot(n, direction) > 0:
                n = -n
                flip_count += 1

            n = n / (np.linalg.norm(n) + 1e-12)

            hits[i] = p
            hit_normals[i] = n
            valid_mask[i] = True

    if flip_count > 0:
        # print(f"    [Debug] Flipped {flip_count} inverted normals out of {np.sum(valid_mask)} hits.")
        pass

    return hits, hit_normals, valid_mask


# -----------------------------------------------------------------------------
# Logic
# -----------------------------------------------------------------------------

def align_toy_to_shoe(shoe, toy, o_ratio, p_ratio):
    print("\n[Alignment Phase]")
    o = get_point_by_ratio(shoe, *o_ratio)
    normal_o = get_normal_at_point(shoe, o)
    p = get_point_by_ratio(toy, *p_ratio)
    normal_p = get_normal_at_point(toy, p)

    print(f"  Target Shoe Anchor: {o}")
    print(f"  Target Shoe Normal: {normal_o}")
    print(f"  Toy Anchor Normal:  {normal_p}")

    # Rotation
    v_from = -normal_p
    v_to = normal_o
    axis = np.cross(v_from, v_to)
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-6:
        axis = axis / axis_len
        angle = np.degrees(np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0)))
        toy.rotate_vector(vector=axis, angle=angle, point=p, inplace=True)
        print(f"  -> Rotated toy by {angle:.2f} degrees")

    # Dynamic Hover Gap
    shoe_diag = np.linalg.norm(np.array(shoe.bounds[1::2]) - np.array(shoe.bounds[0::2]))
    safe_gap = shoe_diag * 0.05
    toy.translate(o - p + normal_o * safe_gap, inplace=True)
    print(f"  -> Translated to position with {safe_gap:.2f} unit hover gap")

    return o, normal_o


def apply_non_rigid_warp(shoe, toy, shoe_normal, offset=0.2,
                         bottom_percent=50,
                         max_samples=600,
                         preserve_volume=True):
    print("\n[Warp Phase]")

    # 0) Resolution
    if toy.n_points < 5000:
        print(f"  -> Subdividing toy (was {toy.n_points} points)")
        toy = toy.subdivide(1, subfilter="loop")

    # Ensure point normals exist for the selection logic
    if "Normals" not in toy.point_data:
        toy.compute_normals(inplace=True, point_normals=True, cell_normals=False)

    # 1) Smoothed Proxy
    b_orig = shoe.bounds
    diag_orig = np.linalg.norm(np.array(b_orig[1::2]) - np.array(b_orig[0::2]))
    shoe_smooth = shoe.copy().smooth(n_iter=10, relaxation_factor=0.1)

    # 2) Identify Bottom Anchors (Normal-Aware)
    # ---------------------------------------------------------
    # Rule 1: Must be in the bottom spatial region (Height Check)
    toy_center = np.mean(toy.points, axis=0)
    height_dots = np.dot(toy.points - toy_center, -shoe_normal)
    height_thresh = np.percentile(height_dots, bottom_percent)

    # Rule 2: Must point DOWN (Normal Check)
    # This excludes the rim edges which are low but point sideways
    normal_dots = np.dot(toy.point_data["Normals"], -shoe_normal)

    # Combine: Low Position AND Facing Down (dot > 0.5 means angle < 60 deg to down vector)
    # Note: shoe_normal is UP. -shoe_normal is DOWN.
    # If toy normal is DOWN, dot(normal, -shoe_normal) should be close to 1.0
    is_bottom_height = height_dots > height_thresh
    is_facing_down = normal_dots > 0.5

    # Anchors are STRICTLY the flat back surface
    cand_idx = np.where(is_bottom_height & is_facing_down)[0]

    # Detail Points are everything else (Top + Rim)
    # This logic forces the rim to be treated as "Detail" to be preserved
    detail_indices = np.setdiff1d(np.arange(toy.n_points), cand_idx)
    # ---------------------------------------------------------

    # Subsample anchors (Smart Sampling for Corners)
    if len(cand_idx) > max_samples:
        rel_pos = toy.points[cand_idx] - toy_center
        proj_down = np.outer(np.dot(rel_pos, shoe_normal), shoe_normal)
        radial_vec = rel_pos - proj_down
        radial_dist = np.linalg.norm(radial_vec, axis=1)

        n_corner = int(max_samples * 0.40)  # Increased corner weight
        n_random = max_samples - n_corner

        sorted_indices_local = np.argsort(radial_dist)[::-1]
        corner_local = sorted_indices_local[:n_corner]
        remaining_local = sorted_indices_local[n_corner:]
        rng = np.random.default_rng(42)
        random_local = rng.choice(remaining_local, size=n_random, replace=False)

        keep_local = np.concatenate([corner_local, random_local])
        anchor_idx = cand_idx[keep_local]
        print(f"  -> Smart Sampling: {n_corner} corners + {n_random} centers.")
    else:
        anchor_idx = cand_idx

    anchor_pts = toy.points[anchor_idx]
    print(f"  -> Anchors: {len(anchor_pts)} points (Backside only). Preserving {len(detail_indices)} detail points.")

    # 3) Project Bottom Anchors
    print("  -> Projecting anchors...")
    hits, hit_normals, valid = project_points_along_vector(shoe_smooth, anchor_pts, -shoe_normal)

    hit_ratio = np.sum(valid) / len(anchor_pts)
    if hit_ratio < 0.5:
        print("[Error] Too many rays missed. Check alignment.")
        return toy

    final_sources = anchor_pts[valid]
    final_hits = hits[valid]
    final_hit_normals = hit_normals[valid]

    # Apply Offset
    final_targets = final_hits + (final_hit_normals * offset)

    # 4) Calculate Deformation
    print("  -> Calculating TPS warp...")
    displacement = final_targets - final_sources

    rbf = RBFInterpolator(
        final_sources,
        displacement,
        kernel="thin_plate_spline",
        smoothing=0.0
    )

    warped = toy.copy()

    if not preserve_volume:
        warped.points += rbf(warped.points)
    else:
        # Volume/Detail Preservation
        # 1. Warp the Anchors (Backside)
        # We must include ALL candidate bottom points in the warp, not just the subsampled ones
        # to ensure the whole back surface moves smoothly
        back_disp = rbf(warped.points[cand_idx])
        warped.points[cand_idx] += back_disp

        # 2. Warp Details (Rim + Top) relative to nearest Backside neighbor
        # This preserves the texture of the rim because it moves rigidly with the back
        import scipy.spatial
        tree = scipy.spatial.cKDTree(toy.points[cand_idx])
        dists, neighbor_ids = tree.query(toy.points[detail_indices])

        neighbor_disp = back_disp[neighbor_ids]
        warped.points[detail_indices] += neighbor_disp

    return warped, cand_idx


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    dayong_dir = cur_dir.parent
    shoe_path = os.path.join(dayong_dir, "scans", "STLs", "shoe.stl")
    toy_path = os.path.join(dayong_dir, "scans", "STLs", "badge.stl")

    if not os.path.exists(shoe_path) or not os.path.exists(toy_path):
        print("Error: Files not found.")
    else:
        shoe_mesh = pv.read(shoe_path)
        toy_mesh = pv.read(toy_path)
        toy_mesh.scale(2, inplace=True)

        # 1. Print Initial Stats
        print("INITIAL GEOMETRY REPORT")
        print_mesh_stats(shoe_mesh, "Shoe")
        print_mesh_stats(toy_mesh, "Toy (Badge)")

        p1 = pv.Plotter()
        p1.add_text("Final Result", font_size=10)
        # p1.add_mesh(shoe_mesh, color="lightblue", opacity=0.8)
        p1.add_mesh(toy_mesh, color="orange")
        p1.show()

        # 2. Alignment
        o, p = align_toy_to_shoe(shoe_mesh, toy_mesh, (0.5, 0.2, 0.9), (0.5, 0.5, 0.0))

        # 3. Warp
        warped_toy, bottom_indices = apply_non_rigid_warp(
            shoe_mesh, toy_mesh, p,
            offset=0.2,
            bottom_percent=50,
            max_samples=800,
            preserve_volume=True
        )

        # 4. Final Quality Report
        print("\n" + "=" * 60)
        print("FINAL QUALITY CHECK")
        print("=" * 60)
        print_mesh_stats(warped_toy, "Warped Toy")

        # Calculate gaps specifically for the bottom surface (the part that should stick)
        # We reuse the bottom_indices found during warp to check only the relevant interface
        # bottom_subset = warped_toy.extract_points(bottom_indices)
        # bottom_subset.compute_implicit_distance(shoe_mesh, inplace=True)
        # gaps = np.abs(bottom_subset.point_data["implicit_distance"])
        #
        # print(f"Interface Fit Metrics (Bottom Surface to Shoe):")
        # print(f"  Max Gap:      {np.max(gaps):.4f} mm")
        # print(f"  Mean Gap:     {np.mean(gaps):.4f} mm")
        # print(f"  Median Gap:   {np.median(gaps):.4f} mm")

        # if np.mean(gaps) < 0.05:
        #     print("  -> STATUS: PASS (Excellent Fit)")
        # elif np.mean(gaps) < 0.2:
        #     print("  -> STATUS: ACCEPTABLE (Good Fit)")
        # else:
        #     print("  -> STATUS: WARNING (Gap too large, check alignment)")

        p4 = pv.Plotter()
        p4.add_text("Final Result", font_size=10)
        p4.add_mesh(shoe_mesh, color="lightblue", opacity=0.8)
        p4.add_mesh(warped_toy, color="orange")
        p4.show()

        # Export
        # print("\n--- Exporting Result ---")
        # output_path = os.path.join(cur_dir, "combined_shoe_toy.stl")
        # combined_mesh = shoe_mesh + warped_toy
        # combined_mesh.save(output_path)
        # print(f"  -> Saved combined mesh to: {output_path}")