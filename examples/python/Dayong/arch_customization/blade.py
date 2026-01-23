import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import os

def alpha_shape_boundary(points, alpha):
    """
    Compute ordered boundary points from alpha shape.
    Returns a continuous, ordered boundary line.
    """
    tri = Delaunay(points)

    # Find all edges and their triangles
    edges = {}
    for simplex_idx, (ia, ib, ic) in enumerate(tri.simplices):
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Calculate circumradius
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0

        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        if area > 0:
            circum_r = a * b * c / (4.0 * area)
        else:
            circum_r = 0

        # Keep edges from triangles within alpha radius
        if circum_r < alpha:
            for edge in [(ia, ib), (ib, ic), (ic, ia)]:
                edge_key = tuple(sorted(edge))
                if edge_key not in edges:
                    edges[edge_key] = []
                edges[edge_key].append(simplex_idx)

    # Find boundary edges (appear only once)
    boundary_edges = [(edge[0], edge[1]) for edge, triangles in edges.items()
                      if len(triangles) == 1]

    if not boundary_edges:
        return np.array([])

    # Build adjacency list
    adjacency = {}
    for i, j in boundary_edges:
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Order boundary points by traversal
    ordered_indices = []
    visited = set()

    # Start from any boundary point
    current = boundary_edges[0][0]
    ordered_indices.append(current)
    visited.add(current)

    # Traverse the boundary
    while len(visited) < len(adjacency):
        neighbors = [n for n in adjacency[current] if n not in visited]
        if not neighbors:
            break
        current = neighbors[0]
        ordered_indices.append(current)
        visited.add(current)

    # Get ordered boundary points
    boundary_points = points[ordered_indices]

    return boundary_points

def resample_boundary_equal_distance(boundary, num_points=1000):
    """
    Resample boundary with equally spaced points.

    Parameters:
    -----------
    boundary : np.ndarray
        Nx2 array of boundary points
    num_points : int
        Number of points in resampled boundary

    Returns:
    --------
    resampled_boundary : np.ndarray
        Resampled boundary with equal spacing
    """
    # Close the boundary loop
    boundary_closed = np.vstack([boundary, boundary[0]])

    # Calculate cumulative distance along boundary
    distances = np.sqrt(np.sum(np.diff(boundary_closed, axis=0) ** 2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

    # Total perimeter length
    total_length = cumulative_distances[-1]

    # Create equally spaced distances
    equal_distances = np.linspace(0, total_length, num_points)

    # Interpolate X and Y coordinates
    interp_x = interp1d(cumulative_distances, boundary_closed[:, 0], kind='linear')
    interp_y = interp1d(cumulative_distances, boundary_closed[:, 1], kind='linear')

    # Sample at equal distances
    resampled_x = interp_x(equal_distances)
    resampled_y = interp_y(equal_distances)

    resampled_boundary = np.column_stack([resampled_x, resampled_y])

    return resampled_boundary[:-1]  # Remove duplicate closing point

def shrink_boundary(boundary, shrink_ratio=0.01):
    """
    Shrink boundary by moving each point toward the centroid.

    Parameters:
    -----------
    boundary : np.ndarray
        Nx2 array of boundary points
    shrink_ratio : float
        Ratio to shrink (0.01 = 1% shrinkage)

    Returns:
    --------
    shrunken_boundary : np.ndarray
        Shrunken boundary points
    """
    # Calculate centroid
    centroid = np.mean(boundary, axis=0)

    # Scale factor (1% shrink = 0.99 scale)
    scale_factor = 1 - shrink_ratio

    # Shrink: move each point toward centroid
    shrunken_boundary = centroid + (boundary - centroid) * scale_factor

    return shrunken_boundary

def orient_normals_consistently_advanced(normals, points, kdtree):
    """
    Advanced normal orientation that works better for curved surfaces.
    Uses nearest neighbor propagation to maintain consistency.
    """
    oriented = normals.copy()
    visited = np.zeros(len(normals), dtype=bool)

    # Start from the point with largest Z coordinate (or you can choose differently)
    start_idx = np.argmax(points[:, 2])
    visited[start_idx] = True

    # Use a queue for breadth-first propagation
    queue = [start_idx]

    while queue:
        current_idx = queue.pop(0)
        current_normal = oriented[current_idx]

        # Find neighbors
        distances, indices = kdtree.query(points[current_idx], k=min(10, len(points)))

        for neighbor_idx in indices:
            if not visited[neighbor_idx]:
                # Flip normal if it points in opposite direction
                if np.dot(oriented[neighbor_idx], current_normal) < 0:
                    oriented[neighbor_idx] = -oriented[neighbor_idx]

                visited[neighbor_idx] = True
                queue.append(neighbor_idx)

    return oriented

def fit_surface_to_knife(knife_points, n_neighbors=20):
    """
    Fit a surface representation to the knife point cloud.
    Improved version for handling complex curved surfaces.

    Args:
        knife_points: Nx3 array of knife point cloud
        n_neighbors: number of neighbors to use for local normal estimation

    Returns:
        kdtree: KDTree for nearest neighbor queries
        normals: Nx3 array of normal vectors at each knife point
    """
    kdtree = KDTree(knife_points)
    normals = np.zeros_like(knife_points)

    # Estimate normal at each point using local PCA
    for i, point in enumerate(knife_points):
        # Find k nearest neighbors - use more for curved surfaces
        k = min(n_neighbors, len(knife_points))
        distances, indices = kdtree.query(point, k=k)
        neighbors = knife_points[indices]

        # Center the neighbors
        centered = neighbors - neighbors.mean(axis=0)

        # PCA to find the normal (smallest principal component)
        pca = PCA(n_components=3)
        pca.fit(centered)

        # Normal is the direction with smallest variance
        normal = pca.components_[-1]
        normals[i] = normal

    # Orient normals consistently using propagation
    normals = orient_normals_consistently_advanced(normals, knife_points, kdtree)

    return kdtree, normals

def classify_point_side_advanced(point, knife_kdtree, knife_points, knife_normals, k=10, adaptive=True):
    """
    Improved classification for curved surfaces using adaptive weighting.

    Args:
        point: 1x3 point to classify
        knife_kdtree: KDTree of knife points
        knife_points: Nx3 knife point cloud
        knife_normals: Nx3 normal vectors at knife points
        k: number of nearest neighbors to consider
        adaptive: use distance-adaptive weighting

    Returns:
        signed distance (positive on one side, negative on other)
    """
    # Find k nearest points on the knife surface
    distances, indices = knife_kdtree.query(point, k=min(k, len(knife_points)))

    # Handle single index case
    if np.isscalar(indices):
        indices = [indices]
        distances = [distances]

    # Adaptive weighting based on distance
    if adaptive:
        # Use Gaussian-like weighting for smoother transitions
        sigma = np.mean(distances) if len(distances) > 1 else distances[0] + 1e-10
        weights = np.exp(-np.array(distances) ** 2 / (2 * sigma ** 2))
    else:
        # Inverse distance weighting
        weights = 1.0 / (np.array(distances) + 1e-10)

    weights /= weights.sum()

    signed_distances = []
    for idx, weight in zip(indices, weights):
        nearest_point = knife_points[idx]
        normal = knife_normals[idx]

        # Vector from knife point to query point
        vec = point - nearest_point

        # Signed distance along normal direction
        signed_dist = np.dot(vec, normal)
        signed_distances.append(signed_dist * weight)

    return sum(signed_distances)

def cut_point_cloud(target_points, knife_points, n_neighbors=20, k_classify=10,
                    adaptive=True, progress_callback=None):
    """
    Cut a target point cloud into two pieces using a knife point cloud.
    Improved version for complex curved surfaces.

    Args:
        target_points: Mx3 array of target point cloud to be cut
        knife_points: Nx3 array of knife point cloud (curved surface)
        n_neighbors: neighbors for normal estimation on knife (increase for smoother surfaces)
        k_classify: neighbors for classification (increase for curved surfaces)
        adaptive: use adaptive weighting for classification
        progress_callback: optional callback function for progress updates

    Returns:
        piece1: Points on positive side of knife
        piece2: Points on negative side of knife
        signed_distances: array of signed distances for each point
    """
    # Fit surface to knife
    print("Fitting surface to knife point cloud...")
    print(f"  Using {n_neighbors} neighbors for normal estimation")
    knife_kdtree, knife_normals = fit_surface_to_knife(knife_points, n_neighbors)

    # Classify each target point
    print("Classifying target points...")
    print(f"  Using {k_classify} neighbors for classification")
    print(f"  Adaptive weighting: {adaptive}")

    signed_distances = np.zeros(len(target_points))

    # Progress tracking
    total = len(target_points)
    progress_interval = max(1, total // 20)  # Report every 5%

    for i, point in enumerate(target_points):
        signed_distances[i] = classify_point_side_advanced(
            point, knife_kdtree, knife_points, knife_normals, k_classify, adaptive
        )

        # Progress update
        if progress_callback and i % progress_interval == 0:
            progress_callback(i, total)
        elif i % progress_interval == 0:
            print(f"  Progress: {i}/{total} ({100 * i / total:.1f}%)")

    # Split into two pieces
    piece1 = target_points[signed_distances >= 0]
    piece2 = target_points[signed_distances < 0]

    print(f"\nCut complete!")
    print(f"  Piece 1: {len(piece1)} points ({len(piece1) / len(target_points) * 100:.1f}%)")
    print(f"  Piece 2: {len(piece2)} points ({len(piece2) / len(target_points) * 100:.1f}%)")

    return piece1, piece2, signed_distances

def call_surface_cut(pcd_target: o3d.geometry.PointCloud, pcd_knife):
    target_points = np.asarray(pcd_target.points)  # generate_curved_surface(n_points=3000, surface_type='complex_target')
    knife_points =  np.asarray(pcd_knife.points)  # generate_curved_surface(n_points=1500, surface_type='wavy_knife')

    print(f"Target points: {len(target_points)}")
    print(f"Knife points: {len(knife_points)}")

    # Perform the cut with parameters optimized for curved surfaces
    piece1, piece2, signed_distances = cut_point_cloud(
        target_points,
        knife_points,
        n_neighbors=25,  # More neighbors for smoother normal estimation
        k_classify=15,  # More neighbors for classification
        adaptive=True  # Use adaptive weighting
    )
    if len(piece1) > len(piece2):
        save_data1 = piece1
    else:
        save_data1 = piece2

    saved_M1 = o3d.geometry.PointCloud()
    saved_M1.points = o3d.utility.Vector3dVector(save_data1)

    return saved_M1

def extract_points_by_y_range(pcd: o3d.geometry.PointCloud, y_min, y_max):
    """
    Extract points within a Y range.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Input point cloud
    y_min : float
        Minimum Y value
    y_max : float
        Maximum Y value

    Returns:
    --------
    filtered_pcd : open3d.geometry.PointCloud
        Point cloud with points in Y range
    """
    # Convert to numpy
    points = np.asarray(pcd.points)

    # Get Y coordinates
    y_values = points[:, 1]

    # Create mask for points within range
    mask = (y_values >= y_min) & (y_values <= y_max)

    # Get indices of points within range
    indices = np.where(mask)[0]

    print(f"Y range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"Points in range: {len(indices)} / {len(points)} ({len(indices) / len(points) * 100:.1f}%)")

    # Extract points
    filtered_pcd = pcd.select_by_index(indices)

    return filtered_pcd

def estimate_eps(pcd: o3d.geometry.PointCloud, k=10):
    """
    Estimate good eps value using k-nearest neighbors.
    """
    points = np.asarray(pcd.points)

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Take mean of k-th nearest neighbor distances
    k_distances = np.sort(distances[:, -1])

    # Use elbow method: take 95th percentile
    suggested_eps = np.percentile(k_distances, 95)

    print(f"Suggested eps value: {suggested_eps:.4f}")

    return suggested_eps

def get_largest_cluster_auto(pcd: o3d.geometry.PointCloud, min_points=10):
    """
    Automatically estimate eps and find largest cluster.
    """
    # Estimate eps
    eps = estimate_eps(pcd, k=min_points)

    print(f"\nUsing eps={eps:.4f}, min_points={min_points}")

    # Cluster
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # Find largest
    unique_labels = labels[labels >= 0]
    if len(unique_labels) == 0:
        print("No clusters found!")
        return pcd

    unique, counts = np.unique(unique_labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]

    print(f"Found {len(unique)} clusters, largest has {counts.max()} points")

    # Extract largest
    largest_pcd = pcd.select_by_index(np.where(labels == largest_label)[0])

    return largest_pcd

def segment_plane_ransac(pcd: o3d.geometry.PointCloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Segment plane using RANSAC algorithm.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Input point cloud
    distance_threshold : float
        Max distance a point can be from the plane to be considered an inlier
    ransac_n : int
        Number of points to sample for generating a plane
    num_iterations : int
        Number of iterations

    Returns:
    --------
    plane_pcd : open3d.geometry.PointCloud
        Points belonging to the plane
    non_plane_pcd : open3d.geometry.PointCloud
        Points not belonging to the plane
    plane_model : list
        Plane equation [a, b, c, d] where ax + by + cz + d = 0
    """
    # Segment the plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    # Extract plane equation
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"Number of inliers (plane points): {len(inliers)}")
    print(f"Number of outliers (non-plane points): {len(pcd.points) - len(inliers)}")

    # Extract plane points (inliers)
    plane_pcd = pcd.select_by_index(inliers)
    plane_pcd.paint_uniform_color([1, 0, 0])  # Red for plane

    # Extract non-plane points (outliers)
    non_plane_pcd = pcd.select_by_index(inliers, invert=True)
    non_plane_pcd.paint_uniform_color([0, 0, 1])  # Blue for non-plane

    return plane_pcd, non_plane_pcd, plane_model

def get_bottom_surface(pcd: o3d.geometry.PointCloud, alpha: float = 5.0, y_min_ratio: float = 0.0, y_max_ratio: float = 1.0):
    # 1. Project the pcd to the XY plane and get the boundary
    xyz_points = pcd.points
    xy_points = np.asarray(xyz_points)[:, 0:2]
    plt.figure(figsize=(10, 8))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=1, c='blue', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')  # Equal aspect ratio
    plt.title('XY Point Cloud')
    plt.show()

    boundary = alpha_shape_boundary(xy_points, alpha)
    plt.figure(figsize=(12, 10))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='lightblue', alpha=0.3, label='All points')
    plt.plot(boundary[:, 0], boundary[:, 1], 'r', linewidth=2, label='Boundary')  ### r-
    plt.plot([boundary[-1, 0], boundary[0, 0]], [boundary[-1, 1], boundary[0, 1]], 'r-', linewidth=2)  # Close the loop
    plt.axis('equal')
    plt.legend()
    plt.title('Continuous Alpha Shape Boundary')
    plt.show()

    print(f"Boundary has {len(boundary)} points")

    # 2. Resample points on the boundary to the same distance
    equal_boundary = resample_boundary_equal_distance(boundary, num_points=1000)

    # Verify spacing
    distances = np.sqrt(np.sum(np.diff(equal_boundary, axis=0) ** 2, axis=1))
    print(f"Mean distance: {distances.mean():.6f}")
    print(f"Std distance: {distances.std():.6f}")
    print(f"Min distance: {distances.min():.6f}")
    print(f"Max distance: {distances.max():.6f}")

    plt.figure(figsize=(12, 10))
    plt.plot(boundary[:, 0], boundary[:, 1], 'b-', alpha=0.3, label='Original')
    plt.scatter(equal_boundary[:, 0], equal_boundary[:, 1], s=2, c='red', label='Equal spacing')
    plt.axis('equal')
    plt.legend()
    plt.title('Resampled Boundary with Equal Spacing')
    plt.show()

    # 3. Shrink the boundary by shrink_ratio
    shrunken = shrink_boundary(equal_boundary, shrink_ratio=0.01)  # 1% or 5% shrink

    plt.figure(figsize=(12, 10))
    plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='lightblue', alpha=0.3, label='Points')
    plt.plot(boundary[:, 0], boundary[:, 1], 'b-', linewidth=2, label='Original boundary')
    plt.plot(shrunken[:, 0], shrunken[:, 1], 'r-', linewidth=2, label='Shrunken 1%')
    plt.plot([shrunken[-1, 0], shrunken[0, 0]],
             [shrunken[-1, 1], shrunken[0, 1]], 'r-', linewidth=2)
    plt.axis('equal')
    plt.legend()
    plt.title('Boundary Shrinkage')
    plt.show()

    # 4. Stack the points of the boundary along the Z axis (i.e. form the blade)
    min_bound = pcd.get_min_bound()
    min_z = min_bound[2]  # Z is the third component
    print(f"Minimum Z value: {min_z}")

    max_z = min_z + 30
    # -------------------- populate the cutting plane -----------------
    # Create multiple shrunken boundaries at different Z heights
    z_layers = []
    num_layers = 30
    z_start = min_z
    z_end = max_z
    z_values = np.linspace(z_start, z_end, num_layers)

    for i, z_value in enumerate(z_values):
        # Create 3D points with current Z value
        shrunken_3d = np.column_stack([shrunken, np.full(len(shrunken), z_value)])
        z_layers.append(shrunken_3d)
        print(f"Layer {i}: Z = {z_value:.4f}, Points = {len(shrunken_3d)}")

    # Stack all layers into one array
    all_layers = np.vstack(z_layers)
    print(f"\nTotal points across all layers: {len(all_layers)}")

    # Create point cloud from all layers
    layers_pcd = o3d.geometry.PointCloud()
    layers_pcd.points = o3d.utility.Vector3dVector(all_layers)

    o3d.visualization.draw_geometries(
        [layers_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
        window_name="Aligned to Y-axis: Stationary",
        width=1400,
        height=900
    )

    # Assign different colors to each point cloud
    pcd1 = pcd
    pcd2 = layers_pcd

    pcd11_vis = pcd1.paint_uniform_color([1, 0, 0])  # Red
    pcd22_vis = pcd2.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize original clouds with box
    o3d.visualization.draw_geometries([pcd11_vis, pcd22_vis],
                                      window_name="Original - Red: pcd1, Blue: pcd2")

    print("\nSurface Processing...")

    # 5. Cut the pcd with the blade
    remain_pcd1 = call_surface_cut(pcd1, pcd2)  ### pcd1: target; pcd2: knift. Point Cloud

    # Get bounding box bounds
    min_bound = pcd1.get_min_bound()
    max_bound = pcd1.get_max_bound()

    min_y = min_bound[1]  # Y is index 1
    max_y = max_bound[1]

    # Usage
    y_min = min_y + (max_y - min_y) * y_min_ratio  # Your minimum Y value 0.01
    y_max = min_y + (max_y - min_y) * y_max_ratio  # Your maximum Y value 0.8

    # 6. First cut the front and back part of the foot (trivial part);
    # then use DBSCAN to get the largest connected component
    cleaned_pcd = extract_points_by_y_range(remain_pcd1, y_min, y_max)
    largest_piece = get_largest_cluster_auto(cleaned_pcd, min_points=50)

    # Visualize original clouds with box
    o3d.visualization.draw_geometries([largest_piece],
                                      window_name="Processed - Blue")
    return largest_piece

    # # 7. Use the largest_piece as the plane; non_plane_pcd contains the arch
    # plane_pcd, non_plane_pcd, plane_model = segment_plane_ransac(
    #     largest_piece,
    #     distance_threshold=1.5,  # Adjust based on your data
    #     ransac_n=3,
    #     num_iterations=200
    # )
    #
    # arch_raw = get_largest_cluster_auto(non_plane_pcd, min_points=50)
    # pcd4_vis = arch_raw.paint_uniform_color([0, 1, 0])  # Green
    #
    # ###   cut the arch region
    # # Get bounding box bounds
    # min_bound = pcd1.get_min_bound()
    # max_bound = pcd1.get_max_bound()
    #
    # min_y = min_bound[1]  # Y is index 1
    # max_y = max_bound[1]
    #
    # y_min = min_y + (max_y - min_y) * 0.2  # Your minimum Y value
    # y_max = min_y + (max_y - min_y) * 0.7  # Your maximum Y value
    #
    # # filtered_pcd is the arch extracted from arch_raw (set some y limits)
    # filtered_pcd = extract_points_by_y_range(arch_raw, y_min, y_max)
    # filtered_pcd = get_largest_cluster_auto(filtered_pcd, min_points=50)
    # pcd5_vis = filtered_pcd.paint_uniform_color([1, 0, 1])  # Magenta
    # o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")
    #
    # return filtered_pcd

def extract_surface_from_grid(
        pcd: o3d.geometry.PointCloud,
        grid_size: float = 2.0,  # mm, grid cell size
        dz_below: float = 0.8,  # mm, keep points up to this much below the upper surface
        dz_above: float = 0.2,  # mm, (optional) allow a bit above due to noise
        min_points_per_cell: int = 1
) -> o3d.geometry.PointCloud:
    """
    Use grid to estimate the upper_z (= max z) if each cell,
    then keep all the points that are within [upper_z - dz_below, upper_z + dz_above].

    Works for both insole and foot.
    """

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError("Empty point cloud")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x0, y0 = x.min(), y.min()
    ix = np.floor((x - x0) / grid_size).astype(np.int32)
    iy = np.floor((y - y0) / grid_size).astype(np.int32)

    big = int(iy.max() + 10)
    key = ix * big + iy

    # sort by key then by z so we can get max z per cell quickly
    order = np.lexsort((z, key))
    key_sorted = key[order]
    z_sorted = z[order]

    change = np.where(np.diff(key_sorted) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(key_sorted)]))

    # cell -> upper_z
    upper_z_map = {}
    valid_keys = set()

    for s, e in zip(starts, ends):
        if e - s < min_points_per_cell:
            continue
        k = int(key_sorted[s])
        upper_z_map[k] = float(z_sorted[e - 1])  # max z in this cell
        valid_keys.add(k)

    # now filter original points by z gate relative to their cell's upper_z
    keep = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        k = int(key[i])
        if k not in valid_keys:
            continue
        uz = upper_z_map[k]
        if (uz - dz_below) <= z[i] <= (uz + dz_above):
            keep[i] = True

    dense_pts = pts[keep]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(dense_pts)
    return out

def clean_pcd_statistical(pcd, nb_neighbors=20, std_ratio=1.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def expand_surface_by_kdtree(
    original_pcd: o3d.geometry.PointCloud,
    seed_surface_pcd: o3d.geometry.PointCloud,
    search_radius: float = 1.2,     # mm, neighborhood radius
    z_tolerance: float = 0.6,      # mm, keep neighbors close in Z to seed points
    max_neighbors: int = 200 # safety cap per seed (avoid huge blow-up)
) -> o3d.geometry.PointCloud:
    """
    Use seed surface points (from grid z-gate) to pull back nearby points
    from the original point cloud using KDTree, with a Z consistency guard.

    Returns a denser surface point cloud.
    """
    P = np.asarray(original_pcd.points)
    S = np.asarray(seed_surface_pcd.points)
    if len(P) == 0 or len(S) == 0:
        raise ValueError("Empty input pcd")

    tree = cKDTree(P[:, :3])

    keep_idx = set()

    # Query neighbors around each seed point
    for s in S:
        idxs = tree.query_ball_point(s, r=search_radius)

        if len(idxs) > max_neighbors:
            # If too many, take closest max_neighbors
            d = np.linalg.norm(P[idxs] - s[None, :], axis=1)
            idxs = list(np.array(idxs)[np.argsort(d)[:max_neighbors]])

        # Z-gate relative to the seed point
        z0 = s[2]
        for i in idxs:
            if abs(P[i, 2] - z0) <= z_tolerance:
                keep_idx.add(i)

    keep_idx = np.array(sorted(list(keep_idx)), dtype=np.int64)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P[keep_idx])
    return out