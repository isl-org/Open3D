import pyvista as pv
import numpy as np
from scipy.interpolate import RBFInterpolator
import os
from pathlib import Path
import scipy.spatial


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


# -----------------------------------------------------------------------------
# Visualization Helper
# -----------------------------------------------------------------------------

def visualize_point_and_normal(plotter: pv.Plotter,
                               point: np.ndarray,
                               normal: np.ndarray,
                               point_color: str = "red",
                               normal_color: str = "red",
                               point_radius: float = 3.0,
                               normal_scale: float = 25.0):
    """Add a debug point + normal arrow (and optional label) into an existing PyVista plotter."""
    p = np.asarray(point, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Point marker
    plotter.add_mesh(pv.Sphere(radius=point_radius, center=p), color=point_color)

    # Normal arrow
    arrow = pv.Arrow(start=p, direction=n, scale=normal_scale)
    plotter.add_mesh(arrow, color=normal_color)


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
    """Return a unit surface normal at a query point.

    Implementation detail:
      - We query the *nearest vertex* to `point` and return its precomputed point normal.
      - This is fast and stable for visualization / small offsets.
      - For normals at arbitrary triangle interior points, we'd need triangle-level normal
        interpolation; here nearest-vertex is sufficient for our use (small offset direction).

    Notes:
      - `auto_orient_normals=True` attempts to make normals globally consistent.
        This is important for STL files where normals can be flipped.
    """
    # Ensure normals exist (point normals are stored in `mesh.point_data['Normals']`).
    if "Normals" not in mesh.point_data:
        mesh.compute_normals(
            inplace=True,
            point_normals=True,
            cell_normals=True,
            auto_orient_normals=True,
        )

    # Find nearest vertex to the query point.
    idx = mesh.find_closest_point(point)

    # Read the point normal and normalize to unit length.
    normal = mesh.point_data["Normals"][idx]
    return normal / (np.linalg.norm(normal) + 1e-12)


def align_toy_to_shoe(shoe, toy, o_ratio, p_ratio):
    print("\n[Alignment Phase]")

    # Make normals consistent (important for arbitrary STLs where normals may be flipped)
    if "Normals" not in shoe.point_data:
        shoe.compute_normals(
            inplace=True,
            point_normals=True,
            cell_normals=True,
            auto_orient_normals=True,
        )
    if "Normals" not in toy.point_data:
        toy.compute_normals(
            inplace=True,
            point_normals=True,
            cell_normals=True,
            auto_orient_normals=True,
        )

    # Anchor points
    o = get_point_by_ratio(shoe, *o_ratio)
    normal_o = get_normal_at_point(shoe, o)
    p = get_point_by_ratio(toy, *p_ratio)
    normal_p = get_normal_at_point(toy, p)
    p_idx = toy.find_closest_point(p)

    print(f"  Target Shoe Anchor: {o}")
    print(f"  Target Shoe Normal: {normal_o}")
    print(f"  Toy Anchor Normal:  {normal_p}")

    # ---------------------------------------------------------------------
    # GOAL: ensure the toy surface normal at point p faces toward -normal_o
    # i.e., after rotation: normal_p_aligned ~= -normal_o
    # ---------------------------------------------------------------------
    target_dir = -normal_o

    # If the toy normal at p points away from the desired direction, flip it.
    # This handles cases where the mesh normals are inverted or the "bottom" is ambiguous.
    if np.dot(normal_p, target_dir) < 0:
        normal_p = -normal_p

    # v_from: current toy normal at p; v_to: desired direction (-shoe normal)
    v_from = normal_p
    v_to = target_dir

    # Robust rotation: handle nearly-parallel and nearly-opposite vectors
    # normalize both vectors
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-12)

    # compute dot product (gives cos(angle))
    dot_v = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))

    if dot_v > 1.0 - 1e-6:
        # Already aligned (angle is 0)
        angle = 0.0
        axis = np.array([1.0, 0.0, 0.0])
    elif dot_v < -1.0 + 1e-6:
        # Opposite direction (180): pick any axis perpendicular to v_from
        # (choose a stable axis by crossing with a basis vector)

        # If vectors are almost opposite, cross(v_from, v_to) would be near zero → unstable.
	    # So you pick an arbitrary basis not parallel to v_from, then cross to get a valid perpendicular axis.
	    # Set angle to exactly 180°.
        basis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(v_from, basis)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, basis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = 180.0
    else:
        # general rotation
        axis = np.cross(v_from, v_to)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = float(np.degrees(np.arccos(dot_v)))

    # Rotates the entire toy around the anchor point p so that the toy stays “pinned” while turning.
    if angle > 1e-5:
        toy.rotate_vector(vector=axis, angle=angle, point=p, inplace=True)
        print(f"  -> Rotated toy by {angle:.2f} degrees (enforce n(p) -> -n(o))")
    else:
        print("  -> Rotation skipped (already aligned)")

    # Dynamic Hover Gap
    shoe_diag = np.linalg.norm(np.array(shoe.bounds[1::2]) - np.array(shoe.bounds[0::2]))
    safe_gap = shoe_diag * 0.12 # safe_gap is 12% of that diagonal: a scale-aware “lift distance”.
    toy.translate(o - p + normal_o * safe_gap, inplace=True)
    print(f"  -> Translated to position with {safe_gap:.2f} unit hover gap")

    # Update toy anchor point/normal after in-place transforms (rotation + translation)
    p_after = toy.points[p_idx]
    normal_p_after = get_normal_at_point(toy, p_after)

    # Return both anchor points and normals (post-transform for toy)
    return o, normal_o, p_after, normal_p_after

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
    shoe_smooth = shoe.copy().smooth(n_iter=10, relaxation_factor=0.1)

    # 2) Identify Bottom Anchors (Normal-Aware)
    # ---------------------------------------------------------
    # Rule 1: Must be in the bottom spatial region (Height Check)
    toy_center = np.mean(toy.points, axis=0)
    height_dots = np.dot(toy.points - toy_center, -shoe_normal) # a scalar per point: how far along that down direction the point is
    height_thresh = np.percentile(height_dots, bottom_percent) # picks a percentile cutoff

    # Rule 2: Must point DOWN (Normal Check)
    # This excludes the rim edges which are low but point sideways
    normal_dots = np.dot(toy.point_data["Normals"], -shoe_normal) # cos(angle) between toy point normal and down direction

    # Combine: Low Position AND Facing Down (dot > 0.5 means angle < 60 deg to down vector)
    # prevents rim edges from being anchors (rim may be low but normals are sideways)
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

    # only ray-project and TPS-fit using anchor_pts (subsampled)
    anchor_pts = toy.points[anchor_idx]
    print(f"  -> Anchors: {len(anchor_pts)} points (Backside only). Preserving {len(detail_indices)} detail points.")

    # 3) Project Bottom Anchors (ray casting)
    '''
    	Cast rays from each anchor point in direction -shoe_normal.
	    Get intersection position (hits) + triangle normal at the hit (hit_normals).
	    Valid indicates which anchors actually hit the shoe.
    '''
    print("  -> Projecting anchors...")
    hits, hit_normals, valid = project_points_along_vector(shoe_smooth, anchor_pts, -shoe_normal)

    hit_ratio = np.sum(valid) / len(anchor_pts)
    if hit_ratio < 0.5:
        print("[Error] Too many rays missed. Check alignment.")
        return toy, cand_idx

    final_sources = anchor_pts[valid]
    final_hits = hits[valid]
    final_hit_normals = hit_normals[valid]

    # Apply Offset
    final_targets = final_hits + (final_hit_normals * offset)

    # 4) Calculate Deformation
    #  final_sources are points on the toy (the anchor points you shot rays from),
    #  and final_targets are the corresponding points on the shoe where those rays hit (plus a small offset)
    # How much should this toy anchor point move (in x,y,z) to land on the shoe surface
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
        '''
            To quickly find, for every “detail/top/rim” point, the nearest “backside/anchor-region” point — 
            so the detail point can “follow” the backside motion without getting TPS-warped itself.
            
            NN search is expensive if you do it naively.
	        Naive approach: for each detail point (maybe 300k points), compare to all backside points (maybe 50k) → O(N·M) (too slow).
	        KD-tree approach: build once O(M log M), then query each detail point O(log M) → fast.
            
        	cand_idx = backside/contact patch (the part you actually warp with TPS)
	        detail_indices = everything else (top + rim + decorative details)
        '''
        import scipy.spatial
        tree = scipy.spatial.cKDTree(toy.points[cand_idx])
        dists, neighbor_ids = tree.query(toy.points[detail_indices])

        neighbor_disp = back_disp[neighbor_ids]
        warped.points[detail_indices] += neighbor_disp

    return warped, cand_idx

# --- Surface closest-point helpers (triangle-surface accurate) ---

def build_surface_locator(polydata: pv.PolyData):
    """Build a fast VTK locator for closest-point queries on a triangle surface.

    Why this exists:
      - `find_closest_point` in PyVista is *vertex*-closest.
      - For refinement we want the closest point on the *triangle surface* (more accurate),
        otherwise thin features may snap to a wrong vertex far away.

    Implementation:
      - `vtkStaticCellLocator` accelerates closest-point-to-triangle queries.
      - It must be built once, then reused for many queries.
    """
    locator = pv._vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    return locator


def closest_points_on_surface(locator, query_points: np.ndarray) -> np.ndarray:
    """Return closest points on the *triangle surface* for each query point.

    Parameters
    ----------
    locator : vtkStaticCellLocator
        Built by `build_surface_locator`.
    query_points : (N,3) float ndarray
        Points we want to pull toward the shoe surface.

    Returns
    -------
    out : (N,3) float ndarray
        Closest point on the surface for each query point.

    Notes
    -----
    - VTK's `FindClosestPoint` is not vectorized, so we loop in Python.
      This is still fast enough for ~1k-10k points (our refine subset).
    """
    out = np.zeros_like(query_points)

    # VTK API expects mutable containers; we reuse them each iteration.
    tmp = [0.0, 0.0, 0.0]
    cell_id = pv._vtk.reference(0)
    sub_id = pv._vtk.reference(0)
    dist2 = pv._vtk.reference(0.0)

    for i, pt in enumerate(query_points):
        # Writes the closest surface point into `tmp`.
        locator.FindClosestPoint(pt, tmp, cell_id, sub_id, dist2)
        out[i] = np.array(tmp, dtype=float)

    return out

def refine_floating_features(
    shoe: pv.PolyData,
    toy_warped: pv.PolyData,
    shoe_normal: np.ndarray,
    tol: float = 0.25,
    max_refine_points: int = 1500,
    offset: float = 0.05,
    smoothing: float = 0.0,
) -> pv.PolyData:
    """Second-pass refinement to pull remaining *floating* thin features onto the shoe.

    Context
    -------
    After the main TPS warp (driven by backside anchors), some thin parts of the toy (tips,
    edges, antennas, swords) may still be "floating" above the shoe because:
      - they are not part of the backside anchor region, and
      - `preserve_volume=True` moves details rigidly with the backside rather than shrink-wrapping.

    High-level idea
    ---------------
    1) Detect vertices still far from the shoe using implicit distance.
    2) Prioritize thin-feature candidates (mesh boundary / low edge-incidence).
    3) For a limited subset of worst offenders, compute the closest point on the shoe surface.
    4) Fit a *local* thin-plate-spline (TPS) displacement field and apply ONLY to those vertices.

    This is intentionally local so we don't distort the whole toy.
    """
    print("\n[Refine Phase]")

    # ------------------------------------------------------------------
    # 0) Use a lightly smoothed shoe as the refinement target.
    #    Smoothing reduces noise so closest-point targets are stable.
    # ------------------------------------------------------------------
    shoe_smooth = shoe.copy().smooth(n_iter=10, relaxation_factor=0.1)

    # Ensure shoe point normals exist (used for a small outward offset).
    if "Normals" not in shoe_smooth.point_data:
        shoe_smooth.compute_normals(
            inplace=True,
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
        )

    # Build a fast locator for closest points on the *triangle surface*.
    locator = build_surface_locator(shoe_smooth)

    # ------------------------------------------------------------------
    # 1) Identify "floating" toy vertices via implicit distance.
    #    `compute_implicit_distance(A)` attaches a scalar to THIS mesh's points
    #    measuring distance to surface A. Values > tol are treated as floating.
    # ------------------------------------------------------------------
    tmp = toy_warped.compute_implicit_distance(shoe_smooth)
    d = tmp.point_data["implicit_distance"]

    floating_idx = np.where(d > tol)[0]
    if floating_idx.size == 0:
        print(f"  -> No floating points > {tol}. Skipping refine.")
        return toy_warped

    # ------------------------------------------------------------------
    # 2) Boundary heuristic to prioritize thin features.
    #    Thin tips/edges often lie on mesh boundaries or have low edge-incidence.
    #    We estimate per-vertex edge incidence using `extract_all_edges()`.
    # ------------------------------------------------------------------
    edges = toy_warped.extract_all_edges()

    if edges.n_points > 0 and edges.lines.size > 0:
        # VTK polylines are stored as: [2, a, b, 2, c, d, ...]
        # Reshape to rows [2, i0, i1] so we can count occurrences of vertex indices.
        seg = edges.lines.reshape(-1, 3)[:, 1:]
        counts = np.bincount(seg.ravel(), minlength=toy_warped.n_points)

        nonzero = counts[counts > 0]
        if nonzero.size > 0:
            # Define "boundary-like" as the bottom 20% of nonzero incidence.
            boundary = counts <= np.percentile(nonzero, 20)
        else:
            boundary = np.zeros(toy_warped.n_points, dtype=bool)
    else:
        boundary = np.zeros(toy_warped.n_points, dtype=bool)

    # Score each vertex by its gap; boundary vertices get a boost so we prefer thin parts.
    scores = d.copy()
    scores[boundary] *= 2.0

    # ------------------------------------------------------------------
    # 3) Pick a limited subset of worst offenders to refine.
    #    We refine only up to `max_refine_points` to keep this local & stable.
    # ------------------------------------------------------------------
    order = np.argsort(-scores[floating_idx])
    floating_idx = floating_idx[order[: min(max_refine_points, floating_idx.size)]]

    # Source points = current positions of selected floating vertices.
    src = toy_warped.points[floating_idx]

    # ------------------------------------------------------------------
    # 4) Compute targets on the shoe: closest surface points + small normal offset.
    # ------------------------------------------------------------------
    hit = closest_points_on_surface(locator, src)

    # Approximate normals at `hit` by querying nearest shoe vertex normal.
    # (Good enough for a small outward offset direction.)
    hit_n = np.array([get_normal_at_point(shoe_smooth, p) for p in hit])

    # Push slightly outward to avoid penetration after snapping.
    tgt = hit + hit_n * offset

    # Displacement vectors for these selected points.
    disp = tgt - src

    print(f"  -> Floating pts selected: {floating_idx.size} (tol={tol}).")
    print("  -> Fitting local TPS warp for refinement...")

    # ------------------------------------------------------------------
    # 5) Fit a local TPS (thin-plate spline) displacement field.
    #    The model maps a 3D position -> 3D displacement.
    # ------------------------------------------------------------------
    rbf = RBFInterpolator(
        src,
        disp,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )

    # ------------------------------------------------------------------
    # 6) Apply ONLY to the selected vertices.
    #    This avoids re-warping the entire toy.
    # ------------------------------------------------------------------
    out = toy_warped.copy()
    out.points[floating_idx] += rbf(out.points[floating_idx])

    # ------------------------------------------------------------------
    # 7) Quick post-check: report remaining gaps for the refined vertices.
    # ------------------------------------------------------------------
    tmp2 = out.compute_implicit_distance(shoe_smooth)
    d2 = tmp2.point_data["implicit_distance"][floating_idx]

    # Clip negative to 0 so "inside" doesn't cancel the average gap.
    mean_gap = float(np.mean(np.clip(d2, 0, None)))
    max_gap = float(np.max(np.clip(d2, 0, None)))
    print(f"  -> After refine: mean_gap={mean_gap:.4f}, max_gap={max_gap:.4f}")

    return out

def visualize_backside_contact(
        shoe: pv.PolyData,
        toy_warped: pv.PolyData,
        cand_idx: np.ndarray,
):
    """Visualize backside/contact points only

    This is a quick qualitative check: we render only the backside points in a single
    color so teammates can visually inspect whether the patch hugs the shoe.
    """
    # Slight smoothing makes the shoe look cleaner, but we are not computing distances here.
    shoe_smooth = shoe.copy().smooth(n_iter=10, relaxation_factor=0.1)

    # Backside/contact point cloud
    pts = toy_warped.points[cand_idx]
    back_cloud = pv.PolyData(pts)

    p = pv.Plotter()
    p.add_text("Backside Contact (Qualitative)", font_size=10)
    p.add_mesh(shoe_smooth, color="lightblue")

    # Render backside points as purple spheres (no scalar bar / legend)
    p.add_mesh(
        back_cloud,
        color="purple",
        render_points_as_spheres=True,
        point_size=6,
    )

    p.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    dayong_dir = cur_dir.parent
    shoe_path = os.path.join(dayong_dir, "scans", "STLs", "Toys", "shoe.stl")
    toy_path = os.path.join(dayong_dir, "scans", "STLs", "Toys", "bitcoin.stl")

    if not os.path.exists(shoe_path) or not os.path.exists(toy_path):
        print("Error: Files not found.")
    else:
        shoe_mesh = pv.read(shoe_path)
        toy_mesh = pv.read(toy_path)
        toy_mesh.scale(0.5, inplace=True)

        # 1. Print Initial Stats
        print("INITIAL GEOMETRY REPORT")
        print_mesh_stats(shoe_mesh, "Shoe")
        print_mesh_stats(toy_mesh, "Toy")

        # 2. Alignment
        o, normal_o, p, normal_p = align_toy_to_shoe(
            shoe_mesh,
            toy_mesh,
            (0.5, 0.2, 0.9),
            (0.5, 0.5, 0.5),
        )

        p1 = pv.Plotter()
        p1.add_text("Pre-warp (Aligned)", font_size=10)
        p1.add_mesh(shoe_mesh, color="lightblue")
        p1.add_mesh(toy_mesh, color="orange")
        # Scale debug gizmos to mesh size
        shoe_diagonal = np.linalg.norm(np.array(shoe_mesh.bounds[1::2]) - np.array(shoe_mesh.bounds[0::2]))
        point_radius = shoe_diagonal * 0.001
        normal_length = shoe_diagonal * 0.08
        # Debug: visualize anchors + normals
        visualize_point_and_normal(
            p1,
            point=o,
            normal=normal_o,
            point_color="yellow",
            normal_color="yellow",
            point_radius=point_radius,
            normal_scale=normal_length,
        )
        visualize_point_and_normal(
            p1,
            point=p,
            normal=normal_p,
            point_color="magenta",
            normal_color="magenta",
            point_radius=point_radius,
            normal_scale=normal_length,
        )
        p1.reset_camera()
        p1.camera.zoom(1.2)
        p1.show()

        # 3. Warp
        warped_toy, bottom_indices = apply_non_rigid_warp(
            shoe_mesh, toy_mesh, normal_o,
            offset=0.0,
            bottom_percent=50,
            max_samples=800,
            preserve_volume=True
        )

        # 4. Refinement: fix thin features that may still be floating (e.g., sword tips)
        warped_toy = refine_floating_features(
            shoe_mesh,
            warped_toy,
            normal_o,
            tol=0.25,        # increase to be less aggressive; decrease to pull more
            max_refine_points=1500, # increase if you have many thin protrusions
            offset=0.05,     # small outward offset to reduce penetration risk
            smoothing=0.0
        )

        # Visualize the attachment quality of the backside points
        visualize_backside_contact(shoe_mesh, warped_toy, bottom_indices)

        # 5. Final Quality Report
        print("\n" + "=" * 60)
        print("FINAL QUALITY CHECK")
        print("=" * 60)
        print_mesh_stats(warped_toy, "Warped Toy")

        p4 = pv.Plotter()
        p4.add_text("Final Result (Warp + Refine)", font_size=10)
        p4.add_mesh(shoe_mesh, color="lightblue")
        p4.add_mesh(warped_toy, color="orange")
        p4.show()

        # Export
        print("\n--- Exporting Result ---")
        output_path = os.path.join(cur_dir, "final.stl")
        combined_mesh = shoe_mesh + warped_toy
        combined_mesh.save(output_path)
        print(f"  -> Saved combined mesh to: {output_path}")