import open3d as o3d
import numpy as np
# from scripts.mesh_process import mesh_measure_methods
# from scripts.mesh_process import mesh_smooth_methods
# from scripts.mesh_process import mesh_common_methods


from open3d.visualization import gui as gui
from open3d.visualization import rendering as rendering
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



### --------------------- Function Region ----------------------

import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def call_surface_cut(pcd_target, pcd_knife):
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



### --------------------- end of cut function ----------------------------------

class PointCloudViewerWithLegend:
    def __init__(self, pcd, distances_mm, window_name="Point Cloud Comparison"):
        self.pcd = pcd
        self.distances_mm = distances_mm
        self.min_dist = distances_mm.min()
        self.max_dist = distances_mm.max()

        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window(window_name, 1600, 900)

        # Create 3D scene widget
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)

        # Setup scene
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        # Create info panel
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # Add title
        title = gui.Label("Distance Analysis (mm)")
        self.panel.add_child(title)
        self.panel.add_fixed(em)

        # Add statistics
        stats_text = self._create_stats_text()
        stats_label = gui.Label(stats_text)
        self.panel.add_child(stats_label)
        self.panel.add_fixed(em)

        # Add color legend
        legend_text = """Color Scale:
Blue   → Small differences
Green  → Medium differences
Yellow → Large differences
Red    → Largest differences"""
        legend_label = gui.Label(legend_text)
        self.panel.add_child(legend_label)

        # Add panel to window
        self.window.add_child(self.panel)

        # Add geometry to scene
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        self.widget3d.scene.add_geometry("point_cloud", self.pcd, mat)

        # Setup camera
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.widget3d.setup_camera(60, bounds, bounds.get_center())
        self.widget3d.scene.set_background([0.1, 0.1, 0.1, 1.0])

        # Set layout callback
        self.window.set_on_layout(self._on_layout)

    def _create_stats_text(self):
        """Create statistics text"""
        return f"""Min:    {self.min_dist:.3f} mm
Max:    {self.max_dist:.3f} mm
Mean:   {self.distances_mm.mean():.3f} mm
Median: {np.median(self.distances_mm):.3f} mm
StdDev: {self.distances_mm.std():.3f} mm"""

    def _on_layout(self, layout_context):
        """Handle window layout"""
        r = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size

        # Panel on the left
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)

        # 3D view takes the rest
        self.widget3d.frame = gui.Rect(r.x + panel_width, r.y,
                                       r.width - panel_width, r.height)

    def run(self):
        """Run the application"""
        gui.Application.instance.run()


def compare_point_clouds_with_legend(pcd1, pcd2, cmap_name='jet'):
    """
    Compare two point clouds and visualize with legend

    Args:
        pcd1: First point cloud or mesh
        pcd2: Second point cloud or mesh
        cmap_name: Colormap name ('jet', 'coolwarm', 'viridis', 'turbo', etc.)
    """
    # Convert meshes to point clouds if needed
    if isinstance(pcd1, o3d.geometry.TriangleMesh):
        print("Converting mesh 1 to point cloud...")
        pcd1_points = pcd1.sample_points_uniformly(number_of_points=50000)
    else:
        pcd1_points = pcd1

    if isinstance(pcd2, o3d.geometry.TriangleMesh):
        print("Converting mesh 2 to point cloud...")
        pcd2_points = pcd2.sample_points_uniformly(number_of_points=50000)
    else:
        pcd2_points = pcd2

    # Compute distances
    print("Computing distances...")
    dists = pcd1_points.compute_point_cloud_distance(pcd2_points)
    dists = np.asarray(dists)

    # Convert to millimeters (adjust multiplier if your units are already in mm)
    dists_mm = dists * 1  # Change to dists * 1 if already in mm

    # Apply colormap
    print("Applying colormap...")
    min_dist = dists_mm.min()
    max_dist = dists_mm.max()
    norm = Normalize(vmin=min_dist, vmax=max_dist)

    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except:
        cmap = plt.cm.get_cmap(cmap_name)

    colors = cmap(norm(dists_mm))[:, :3]  # Remove alpha channel
    pcd1_points.colors = o3d.utility.Vector3dVector(colors)

    # Print statistics
    print("\n" + "=" * 60)
    print("DISTANCE STATISTICS (mm)")
    print("=" * 60)
    print(f"Min:           {min_dist:.3f} mm")
    print(f"Max:           {max_dist:.3f} mm")
    print(f"Mean:          {dists_mm.mean():.3f} mm")
    print(f"Median:        {np.median(dists_mm):.3f} mm")
    print(f"Std Deviation: {dists_mm.std():.3f} mm")
    print(f"95th %ile:     {np.percentile(dists_mm, 95):.3f} mm")
    print("=" * 60 + "\n")

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(dists_mm, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Distance (mm)', fontsize=12)
    plt.ylabel('Number of Points', fontsize=12)
    plt.title('Distribution of Point-to-Point Distances', fontsize=14)
    plt.axvline(dists_mm.mean(), color='red', linestyle='--',
                label=f'Mean: {dists_mm.mean():.3f} mm')
    plt.axvline(np.median(dists_mm), color='green', linestyle='--',
                label=f'Median: {np.median(dists_mm):.3f} mm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create and show viewer with legend
    print("Opening viewer...")
    viewer = PointCloudViewerWithLegend(pcd1_points, dists_mm)
    viewer.run()


def simple_comparison_with_stats(pcd1, pcd2, cmap_name='jet'):
    """
    Simpler alternative using standard Open3D viewer with printed stats
    """
    # Convert meshes to point clouds if needed
    if isinstance(pcd1, o3d.geometry.TriangleMesh):
        print("Converting mesh 1 to point cloud...")
        pcd1_points = pcd1.sample_points_uniformly(number_of_points=50000)
    else:
        pcd1_points = pcd1

    if isinstance(pcd2, o3d.geometry.TriangleMesh):
        print("Converting mesh 2 to point cloud...")
        pcd2_points = pcd2.sample_points_uniformly(number_of_points=50000)
    else:
        pcd2_points = pcd2

    # Compute distances
    print("Computing distances...")
    dists = pcd1_points.compute_point_cloud_distance(pcd2_points)
    dists = np.asarray(dists)
    dists_mm = dists * 1  # Adjust if needed

    # Apply colormap
    min_dist = dists_mm.min()
    max_dist = dists_mm.max()
    norm = Normalize(vmin=min_dist, vmax=max_dist)

    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except:
        cmap = plt.cm.get_cmap(cmap_name)

    colors = cmap(norm(dists_mm))[:, :3]
    pcd1_points.colors = o3d.utility.Vector3dVector(colors)

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("DISTANCE STATISTICS (mm)")
    print("=" * 60)
    print(f"Min:           {min_dist:.3f} mm  (Blue in visualization)")
    print(f"25th %ile:     {np.percentile(dists_mm, 25):.3f} mm")
    print(f"Median:        {np.median(dists_mm):.3f} mm  (Cyan/Green)")
    print(f"Mean:          {dists_mm.mean():.3f} mm")
    print(f"75th %ile:     {np.percentile(dists_mm, 75):.3f} mm")
    print(f"95th %ile:     {np.percentile(dists_mm, 95):.3f} mm")
    print(f"Max:           {max_dist:.3f} mm  (Red in visualization)")
    print(f"Std Deviation: {dists_mm.std():.3f} mm")
    print("=" * 60)
    print(f"\nColor Scale ({cmap_name}):")
    print("  Blue   → Smallest differences")
    print("  Cyan   → Small-medium differences")
    print("  Green  → Medium differences")
    print("  Yellow → Medium-large differences")
    print("  Red    → Largest differences")
    print("=" * 60 + "\n")

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd1_points],
        window_name=f"Difference Map: {min_dist:.3f} - {max_dist:.3f} mm",
        width=1400,
        height=900,
        point_show_normal=False
    )

def overlay_meshes(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    alpha: float = 0.15,
    zmin_target: float = 0.0,
    copy_inputs: bool = True,
):
    """
    Overlay two meshes: center-align in XY (both sent to origin),
    then shift each along Z so their minimum Z equals zmin_target (default 0.0).
    Mesh A is rendered semi-transparent.

    Parameters
    ----------
    mesh_a, mesh_b : TriangleMesh
        Already-read meshes.
    alpha : float
        Transparency for mesh A (0..1).
    zmin_target : float
        Target baseline for minimum Z of both meshes (e.g., 0.0).
    copy_inputs : bool
        If True, operate on clones (non-destructive).
    """
    A = mesh_a.clone() if copy_inputs else mesh_a
    B = mesh_b.clone() if copy_inputs else mesh_b

    # Ensure normals for shading
    if not A.has_vertex_normals(): A.compute_vertex_normals()
    if not B.has_vertex_normals(): B.compute_vertex_normals()

    # 1) Center-align: move both centroids to origin (keeps XY & Z centroids matched initially)
    A.translate(-A.get_center())
    B.translate(-B.get_center())

    # 2) Equalize Z-min baseline (shift along Z only)
    def zmin(mesh):
        return float(np.asarray(mesh.vertices)[:, 2].min())

    # let both to share the lower of their two current Z-min values
    target = min(zmin(A), zmin(B))
    A.translate((0, 0, target - zmin(A)))
    B.translate((0, 0, target - zmin(B)))
    # A.translate((0.0, 0.0, zmin_target - zmin(A)))
    # B.translate((0.0, 0.0, zmin_target - zmin(B)))

    # Optional: color tint for contrast
    A.paint_uniform_color([0.2, 1.0, 1.0])  # bluish
    B.paint_uniform_color([0.3, 0.5, 1.0] )  # reddish [1.0, 0.3, 0.2]

    # Materials (robust to Open3D versions missing AlphaMode)
    matA = o3d.visualization.rendering.MaterialRecord()
    matB = o3d.visualization.rendering.MaterialRecord()
    matA.shader = "defaultLitTransparency" if alpha < 1.0 else "defaultLit"
    matB.shader = "defaultLit"
    matA.base_color = (1.0, 1.0, 1.0, float(alpha))
    matB.base_color = (1.0, 1.0, 1.0, 1.0)
    AlphaMode = getattr(o3d.visualization.rendering, "AlphaMode", None)
    if AlphaMode is not None:
        matA.alpha_mode = AlphaMode.Blend if alpha < 1.0 else AlphaMode.Opaque
        matB.alpha_mode = AlphaMode.Opaque

    # Visualize
    o3d.visualization.draw(
        [
            {"name": "Mesh A (transparent)", "geometry": A, "material": matA},
            {"name": "Mesh B",               "geometry": B, "material": matB},
            {"name": "Axes",                 "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)},
        ],
        show_skybox=False,
        bg_color=(1, 1, 1, 1),
        title=f"Overlayed Meshes (XY centers aligned, Zmin = {zmin_target})"
    )

# Example:
# A = o3d.io.read_triangle_mesh("a.stl")
# B = o3d.io.read_triangle_mesh("b.stl")
# overlay_meshes_with_shared_zmin(A, B, alpha=0.2, zmin_target=0.0)


#### method 2
def compare_meshes(mesh1, mesh2):
    """Compare two meshes by overlaying with alignment and transparency.

    Args:
        mesh1: open3d.geometry.TriangleMesh object
        mesh2: open3d.geometry.TriangleMesh object
    """

    # Copy meshes to avoid modifying originals
    m1 = mesh1.__copy__()
    m2 = mesh2.__copy__()

    # Compute and align centers
    c1 = m1.get_center()
    c2 = m2.get_center()
    m1.translate(-c1)
    m2.translate(-c2)

    # Align minimum z values (base alignment)
    z_min1 = np.asarray(m1.vertices)[:, 2].min()
    z_min2 = np.asarray(m2.vertices)[:, 2].min()
    z_offset = z_min2 - z_min1
    m1.translate([0, 0, z_offset])

    # Set colors and transparency
    # Mesh 1: semi-transparent grey
    m1.paint_uniform_color([0.5, 0.5, 0.5])
    m1.compute_vertex_normals()

    # Mesh 2: solid blue
    m2.paint_uniform_color([0.3, 0.5, 1.0])
    m2.compute_vertex_normals()

    # Visualize with custom render options for transparency
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh Comparison")
    vis.add_geometry(m1)
    vis.add_geometry(m2)

    # Set render options for better transparency
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    # Update view and apply transparency by modifying point size/line width
    vis.poll_events()
    vis.update_renderer()

    # Note: For true transparency in Open3D Visualizer, convert to point cloud
    # or use open3d.visualization.draw_geometries with custom rendering
    print("Note: Mesh 1 (blue) has reduced color intensity for comparison")
    print("For better transparency, consider using point cloud representation")

    vis.run()
    vis.destroy_window()


# Alternative function with better transparency using point clouds
def compare_meshes_transparent(mesh1, mesh2, alpha=0.3):
    """Compare meshes with better transparency using combined rendering.

    Args:
        mesh1: open3d.geometry.TriangleMesh object (will be more transparent)
        mesh2: open3d.geometry.TriangleMesh object (solid)
        alpha: transparency level for mesh1 (0=invisible, 1=opaque)
    """
    # Copy meshes
    m1 = mesh1.__copy__()
    m2 = mesh2.__copy__()

    # Align centers
    c1 = m1.get_center()
    c2 = m2.get_center()
    m1.translate(-c1)
    m2.translate(-c2)

    # Align minimum z values
    x_min1 = np.asarray(m1.vertices)[:, 0].mean(axis=0) # .min()
    x_min2 = np.asarray(m2.vertices)[:, 0].mean(axis=0)
    x_offset = x_min2 - x_min1

    y_min1 = np.asarray(m1.vertices)[:, 1].min()
    y_min2 = np.asarray(m2.vertices)[:, 1].min()
    y_offset = y_min2 - y_min1

    z_min1 = np.asarray(m1.vertices)[:, 2].min()
    z_min2 = np.asarray(m2.vertices)[:, 2].min()
    z_offset = z_min2 - z_min1
    m1.translate([x_offset, y_offset, z_offset])

    # Convert mesh1 to point cloud for transparency effect
    pcd1 = m1.sample_points_uniformly(number_of_points=6000)
    pcd1.paint_uniform_color([0.5, 0.5, 0.5])

    # Keep mesh2 as solid mesh
    m2.paint_uniform_color([0.3, 0.5, 1.0])
    m2.compute_vertex_normals()

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd1, m2],
        window_name="Mesh Comparison (Transparent)",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

#
# import open3d as o3d
# import numpy as np


def get_z_range(pcd):
    """
    Get the min and max Z values of a point cloud

    Args:
        pcd: Open3D PointCloud or TriangleMesh

    Returns:
        tuple: (min_z, max_z)
    """
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        points = np.asarray(pcd.vertices)
    else:
        points = np.asarray(pcd.points)

    z_values = points[:, 2]
    min_z = z_values.min()
    max_z = z_values.max()

    return min_z, max_z


def filter_pointcloud_by_z(pcd, z_min=None, z_max=None):
    """
    Filter point cloud to keep only points within Z range

    Args:
        pcd: Open3D PointCloud or TriangleMesh
        z_min: Minimum Z value (if None, uses point cloud's min Z)
        z_max: Maximum Z value (if None, keeps all points above z_min)

    Returns:
        Filtered point cloud
    """
    # Convert mesh to point cloud if needed
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        # Sample points from mesh
        pcd_points = pcd.sample_points_uniformly(number_of_points=50000)
    else:
        pcd_points = pcd

    # Get points as numpy array
    points = np.asarray(pcd_points.points)
    z_values = points[:, 2]

    # Get current min/max if not specified
    current_min_z = z_values.min()
    current_max_z = z_values.max()

    if z_min is None:
        z_min = current_min_z
    if z_max is None:
        z_max = current_max_z

    print(f"Original point cloud Z range: {current_min_z:.3f} to {current_max_z:.3f}")
    print(f"Filtering to Z range: {z_min:.3f} to {z_max:.3f}")

    # Create mask for points within Z range
    mask = (z_values >= z_min) & (z_values <= z_max)
    indices = np.where(mask)[0]

    # Filter the point cloud
    filtered_pcd = pcd_points.select_by_index(indices)

    print(f"Original points: {len(points)}")
    print(f"Filtered points: {len(filtered_pcd.points)}")
    print(f"Removed: {len(points) - len(filtered_pcd.points)} points")

    return filtered_pcd


def filter_from_min_z_with_distance(pcd, distance_from_min):
    """
    Filter point cloud to keep only points from min Z to (min Z + distance)

    Args:
        pcd: Open3D PointCloud or TriangleMesh
        distance_from_min: Distance above minimum Z to keep

    Returns:
        Filtered point cloud
    """
    # Get min Z value
    min_z, max_z = get_z_range(pcd)

    print(f"\nPoint cloud Z range: {min_z:.3f} to {max_z:.3f}")
    print(f"Keeping points from Z = {min_z:.3f} to Z = {min_z + distance_from_min:.3f}")
    print(f"Distance from min: {distance_from_min:.3f}\n")

    # Filter with the calculated range
    filtered_pcd = filter_pointcloud_by_z(pcd, z_min=min_z, z_max=min_z + distance_from_min)

    return filtered_pcd


def visualize_filtered_comparison(original_pcd, filtered_pcd):
    """
    Visualize original and filtered point clouds side by side

    Args:
        original_pcd: Original point cloud
        filtered_pcd: Filtered point cloud
    """
    # Color them differently
    original_colored = o3d.geometry.PointCloud(original_pcd)
    original_colored.paint_uniform_color([0.5, 0.5, 0.5])  # Gray

    filtered_colored = o3d.geometry.PointCloud(filtered_pcd)
    filtered_colored.paint_uniform_color([1, 0, 0])  # Red

    # Show both together
    print("\nVisualizing: Gray = original, Red = filtered portion")
    o3d.visualization.draw_geometries(
        [original_colored, filtered_colored],
        window_name="Original (Gray) vs Filtered (Red)",
        width=1400,
        height=900
    )

    # Show filtered only
    filtered_colored.paint_uniform_color([0, 0.7, 1])  # Blue
    o3d.visualization.draw_geometries(
        [filtered_colored],
        window_name="Filtered Point Cloud Only",
        width=1400,
        height=900
    )


def get_z_statistics(pcd):
    """
    Print detailed Z-axis statistics for a point cloud

    Args:
        pcd: Open3D PointCloud or TriangleMesh
    """
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        points = np.asarray(pcd.vertices)
    else:
        points = np.asarray(pcd.points)

    z_values = points[:, 2]

    print("\n" + "=" * 60)
    print("Z-AXIS STATISTICS")
    print("=" * 60)
    print(f"Min Z:         {z_values.min():.6f}")
    print(f"Max Z:         {z_values.max():.6f}")
    print(f"Mean Z:        {z_values.mean():.6f}")
    print(f"Median Z:      {np.median(z_values):.6f}")
    print(f"Std Dev:       {z_values.std():.6f}")
    print(f"Z Range:       {z_values.max() - z_values.min():.6f}")
    print(f"25th %ile:     {np.percentile(z_values, 25):.6f}")
    print(f"75th %ile:     {np.percentile(z_values, 75):.6f}")
    print("=" * 60 + "\n")


# # # Example usage:
# # if __name__ == "__main__":
# #     # Example 1: Create a test point cloud (cylinder)
# #     print("Creating test point cloud...")
# #     mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=5.0)
# #     mesh.compute_vertex_normals()
# #     pcd = mesh.sample_points_uniformly(number_of_points=10000)
# #
# #     # Get Z statistics
# #     get_z_statistics(pcd)
# #
# #     # Method 1: Filter from min Z with specific distance
# #     print("\n--- Method 1: Filter from min Z + distance ---")
# #     distance = 2.0  # Keep 2.0 units from the bottom
# #     filtered_pcd = filter_from_min_z_with_distance(pcd, distance_from_min=distance)
# #
# #     # Visualize
# #     visualize_filtered_comparison(pcd, filtered_pcd)
# #
# #     # Method 2: Filter with custom Z range
# #     print("\n--- Method 2: Custom Z range ---")
# #     min_z, max_z = get_z_range(pcd)
# #     custom_filtered = filter_pointcloud_by_z(pcd, z_min=min_z, z_max=min_z + 1.5)
# #
# #     # Method 3: Filter by percentages
# #     print("\n--- Method 3: Keep bottom 40% of point cloud ---")
# #     min_z, max_z = get_z_range(pcd)
# #     z_range = max_z - min_z
# #     percentage_filtered = filter_pointcloud_by_z(pcd, z_min=min_z, z_max=min_z + 0.4 * z_range)
# #
# #     breakpoint()
#
#     # Save filtered point cloud
#     # o3d.io.write_point_cloud("filtered_pointcloud.ply", filtered_pcd)
#     # print("Saved filtered point cloud to 'filtered_pointcloud.ply'")
#
#     # For your own point cloud:
#     """
#     # Load your point cloud
#     my_pcd = o3d.io.read_point_cloud("your_pointcloud.ply")
#
#     # Get Z range
#     min_z, max_z = get_z_range(my_pcd)
#     print(f"Z range: {min_z} to {max_z}")
#
#     # Filter: keep from min Z to (min Z + 10mm)
#     filtered = filter_from_min_z_with_distance(my_pcd, distance_from_min=10.0)
#
#     # Visualize
#     visualize_filtered_comparison(my_pcd, filtered)
#
#     # Save result
#     o3d.io.write_point_cloud("filtered_result.ply", filtered)
#     """

import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def compute_pca_orientation(pcd):
    """
    Compute principal axes of a point cloud using PCA

    Args:
        pcd: Open3D PointCloud or TriangleMesh

    Returns:
        dict containing:
            - center: centroid of the point cloud
            - eigenvectors: principal axes (columns are the axes)
            - eigenvalues: variance along each axis
            - angles: rotation angles (in degrees) relative to world axes
    """
    # Get points
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        points = np.asarray(pcd.vertices)
    else:
        points = np.asarray(pcd.points)

    # Compute centroid
    center = points.mean(axis=0)

    # Center the points
    centered_points = points - center

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # Get principal components (eigenvectors)
    eigenvectors = pca.components_.T  # Each column is a principal axis
    eigenvalues = pca.explained_variance_

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Compute rotation angles (in degrees)
    # Angles between principal axes and world axes
    angles = {}
    axis_names = ['X', 'Y', 'Z']
    world_axes = np.eye(3)

    for i, name in enumerate(axis_names):
        for j, world_name in enumerate(axis_names):
            angle = np.arccos(np.clip(np.dot(eigenvectors[:, i], world_axes[:, j]), -1, 1))
            angles[f'PC{i + 1}_to_{world_name}'] = np.degrees(angle)

    # Print results
    print("\n" + "=" * 70)
    print("PCA ORIENTATION ANALYSIS")
    print("=" * 70)
    print(f"\nCentroid: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"\nEigenvalues (variance along each axis):")
    print(f"  PC1 (major): {eigenvalues[0]:.6f} ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
    print(f"  PC2 (mid):   {eigenvalues[1]:.6f} ({pca.explained_variance_ratio_[1] * 100:.2f}%)")
    print(f"  PC3 (minor): {eigenvalues[2]:.6f} ({pca.explained_variance_ratio_[2] * 100:.2f}%)")

    print(f"\nPrincipal Axes (eigenvectors):")
    print(f"  PC1: [{eigenvectors[0, 0]:+.4f}, {eigenvectors[1, 0]:+.4f}, {eigenvectors[2, 0]:+.4f}]")
    print(f"  PC2: [{eigenvectors[0, 1]:+.4f}, {eigenvectors[1, 1]:+.4f}, {eigenvectors[2, 1]:+.4f}]")
    print(f"  PC3: [{eigenvectors[0, 2]:+.4f}, {eigenvectors[1, 2]:+.4f}, {eigenvectors[2, 2]:+.4f}]")
    print("=" * 70 + "\n")

    return {
        'center': center,
        'eigenvectors': eigenvectors,
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'angles': angles
    }


def visualize_pca_axes(pcd, pca_result, axis_length=None):
    """
    Visualize point cloud with PCA axes

    Args:
        pcd: Original point cloud
        pca_result: Result from compute_pca_orientation()
        axis_length: Length of axes to draw (auto-calculated if None)
    """
    # Convert mesh to point cloud if needed
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd_vis = pcd.sample_points_uniformly(number_of_points=10000)
    else:
        pcd_vis = o3d.geometry.PointCloud(pcd)

    pcd_vis.paint_uniform_color([0.7, 0.7, 0.7])

    center = pca_result['center']
    eigenvectors = pca_result['eigenvectors']
    eigenvalues = pca_result['eigenvalues']

    # Auto-calculate axis length based on eigenvalues
    if axis_length is None:
        axis_length = 2 * np.sqrt(eigenvalues[0])

    # Create coordinate frame at centroid
    geometries = [pcd_vis]

    # Create axes as line sets with different colors
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue
    labels = ['PC1 (Major)', 'PC2 (Mid)', 'PC3 (Minor)']

    for i in range(3):
        # Create line for each axis
        axis_points = [center, center + eigenvectors[:, i] * axis_length]
        lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(axis_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([colors[i]])
        geometries.append(line_set)

        # Add sphere at the end of each axis
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.05)
        sphere.translate(center + eigenvectors[:, i] * axis_length)
        sphere.paint_uniform_color(colors[i])
        geometries.append(sphere)

    # Add centroid marker
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * 0.08)
    centroid_sphere.translate(center)
    centroid_sphere.paint_uniform_color([1, 1, 0])  # Yellow
    geometries.append(centroid_sphere)

    # Add world coordinate frame for reference
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_length * 0.5, origin=[0, 0, 0])
    geometries.append(world_frame)

    print("Visualization Legend:")
    print("  Red axis   = PC1 (Major axis, largest variance)")
    print("  Green axis = PC2 (Mid axis)")
    print("  Blue axis  = PC3 (Minor axis, smallest variance)")
    print("  Yellow     = Centroid")
    print("  RGB frame  = World coordinate system")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="PCA Orientation Analysis",
        width=1400,
        height=900
    )


def align_pointcloud_to_pca(pcd, pca_result):
    """
    Align point cloud so that principal axes match world axes

    Args:
        pcd: Original point cloud
        pca_result: Result from compute_pca_orientation()

    Returns:
        Aligned point cloud
    """
    # Convert mesh to point cloud if needed
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd_aligned = pcd.sample_points_uniformly(number_of_points=50000)
    else:
        pcd_aligned = o3d.geometry.PointCloud(pcd)

    center = pca_result['center']
    eigenvectors = pca_result['eigenvectors']

    # Create transformation matrix
    # Rotation: eigenvectors form rotation matrix (transpose to get inverse)
    R = eigenvectors.T

    # Translation: move centroid to origin
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center

    # Apply transformation
    pcd_aligned.transform(T)

    print("Point cloud aligned to PCA axes")
    print("PC1 → X-axis, PC2 → Y-axis, PC3 → Z-axis")

    return pcd_aligned


# def align_major_axis_to_axis(pcd, target_axis='y', pca_result=None):
#     """
#     Align the major principal axis (PC1) to a specific world axis
#
#     Args:
#         pcd: Original point cloud
#         target_axis: Target axis ('x', 'y', or 'z')
#         pca_result: Pre-computed PCA result (will compute if None)
#
#     Returns:
#         Aligned point cloud
#     """
#     # Convert mesh to point cloud if needed
#     if isinstance(pcd, o3d.geometry.TriangleMesh):
#         pcd_aligned = pcd.sample_points_uniformly(number_of_points=50000)
#     else:
#         pcd_aligned = o3d.geometry.PointCloud(pcd)
#
#     # Compute PCA if not provided
#     if pca_result is None:
#         pca_result = compute_pca_orientation(pcd)
#
#     center = pca_result['center']
#     eigenvectors = pca_result['eigenvectors']
#     major_axis = eigenvectors[:, 0]  # PC1 - major axis
#
#     # Define target direction
#     target_vectors = {
#         'x': np.array([1, 0, 0]),
#         'y': np.array([0, 1, 0]),
#         'z': np.array([0, 0, 1])
#     }
#     target_vec = target_vectors[target_axis.lower()]
#
#     # Compute rotation to align major_axis with target_vec
#     # Using Rodrigues' rotation formula
#     v = np.cross(major_axis, target_vec)
#     s = np.linalg.norm(v)
#     c = np.dot(major_axis, target_vec)
#
#     if s < 1e-6:  # Already aligned or opposite
#         if c > 0:  # Already aligned
#             R = np.eye(3)
#         else:  # Opposite direction, rotate 180 degrees
#             # Find perpendicular axis for 180 degree rotation
#             if abs(major_axis[0]) < 0.9:
#                 perp = np.array([1, 0, 0])
#             else:
#                 perp = np.array([0, 1, 0])
#             perp = perp - np.dot(perp, major_axis) * major_axis
#             perp = perp / np.linalg.norm(perp)
#             R = 2 * np.outer(perp, perp) - np.eye(3)
#     else:
#         # Rotation matrix using Rodrigues' formula
#         vx = np.array([[0, -v[2], v[1]],
#                        [v[2], 0, -v[0]],
#                        [-v[1], v[0], 0]])
#         R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
#
#     # Create full transformation matrix
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = -R @ center
#
#     # Apply transformation
#     pcd_aligned.transform(T)
#
#     print(f"Point cloud aligned: Major axis (PC1) → {target_axis.upper()}-axis")
#
#     return pcd_aligned

def align_major_axis_to_axis(pcd, target_axis='y', pca_result=None):
    """
    Align the major principal axis (PC1) to a specific world axis

    Args:
        pcd: Original point cloud
        target_axis: Target axis ('x', 'y', or 'z')
        pca_result: Pre-computed PCA result (will compute if None)

    Returns:
        tuple: (pcd_aligned, T)
            - pcd_aligned: aligned point cloud
            - T: 4x4 transform applied to pcd_aligned (so you can apply the same T to a mesh)
    """
    # Convert mesh to point cloud if needed
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd_aligned = pcd.sample_points_uniformly(number_of_points=50000)
    else:
        pcd_aligned = o3d.geometry.PointCloud(pcd)

    # Compute PCA if not provided
    if pca_result is None:
        pca_result = compute_pca_orientation(pcd)

    center = pca_result['center']
    eigenvectors = pca_result['eigenvectors']
    major_axis = eigenvectors[:, 0]  # PC1 - major axis

    # Define target direction
    target_vectors = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    target_vec = target_vectors[target_axis.lower()]

    # Compute rotation to align major_axis with target_vec
    # Using Rodrigues' rotation formula
    v = np.cross(major_axis, target_vec)
    s = np.linalg.norm(v)
    c = np.dot(major_axis, target_vec)

    if s < 1e-6:  # Already aligned or opposite
        if c > 0:  # Already aligned
            R = np.eye(3)
        else:  # Opposite direction, rotate 180 degrees
            # Find perpendicular axis for 180 degree rotation
            if abs(major_axis[0]) < 0.9:
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            perp = perp - np.dot(perp, major_axis) * major_axis
            perp = perp / np.linalg.norm(perp)
            R = 2 * np.outer(perp, perp) - np.eye(3)
    else:
        # Rotation matrix using Rodrigues' formula
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # Create full transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center

    # Apply transformation
    pcd_aligned.transform(T)

    print(f"Point cloud aligned: Major axis (PC1) → {target_axis.upper()}-axis")

    return pcd_aligned, T


def align_axis_to_axis(pcd, source_axis='pc1', target_axis='y', pca_result=None):
    """
    Align any principal axis to any world axis with more control

    Args:
        pcd: Original point cloud
        source_axis: Which PC axis to align ('pc1', 'pc2', or 'pc3')
        target_axis: Target world axis ('x', 'y', or 'z')
        pca_result: Pre-computed PCA result (will compute if None)

    Returns:
        Aligned point cloud
    """
    # Convert mesh to point cloud if needed
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd_aligned = pcd.sample_points_uniformly(number_of_points=50000)
    else:
        pcd_aligned = o3d.geometry.PointCloud(pcd)

    # Compute PCA if not provided
    if pca_result is None:
        pca_result = compute_pca_orientation(pcd)

    center = pca_result['center']
    eigenvectors = pca_result['eigenvectors']

    # Select source axis
    source_idx = {'pc1': 0, 'pc2': 1, 'pc3': 2}[source_axis.lower()]
    source_vec = eigenvectors[:, source_idx]

    # Define target direction
    target_vectors = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    target_vec = target_vectors[target_axis.lower()]

    # Compute rotation
    v = np.cross(source_vec, target_vec)
    s = np.linalg.norm(v)
    c = np.dot(source_vec, target_vec)

    if s < 1e-6:
        if c > 0:
            R = np.eye(3)
        else:
            if abs(source_vec[0]) < 0.9:
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            perp = perp - np.dot(perp, source_vec) * source_vec
            perp = perp / np.linalg.norm(perp)
            R = 2 * np.outer(perp, perp) - np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center

    # Apply transformation
    pcd_aligned.transform(T)

    print(f"Point cloud aligned: {source_axis.upper()} → {target_axis.upper()}-axis")

    return pcd_aligned


def get_orientation_angle(pcd, reference_axis='z'):
    """
    Get the angle between the major principal axis and a reference world axis

    Args:
        pcd: Point cloud
        reference_axis: 'x', 'y', or 'z'

    Returns:
        Angle in degrees
    """
    pca_result = compute_pca_orientation(pcd)
    major_axis = pca_result['eigenvectors'][:, 0]

    reference_vectors = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }

    ref_vec = reference_vectors[reference_axis.lower()]

    # Compute angle
    cos_angle = np.dot(major_axis, ref_vec)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # Take the acute angle
    if angle > 90:
        angle = 180 - angle

    print(f"\nAngle between major axis (PC1) and {reference_axis.upper()}-axis: {angle:.2f}°")

    return angle


def compare_orientations(pcd1, pcd2):
    """
    Compare orientations of two point clouds using PCA

    Args:
        pcd1: First point cloud
        pcd2: Second point cloud

    Returns:
        Angle difference in degrees
    """
    print("\n--- Point Cloud 1 ---")
    pca1 = compute_pca_orientation(pcd1)

    print("\n--- Point Cloud 2 ---")
    pca2 = compute_pca_orientation(pcd2)

    # Compute angle between major axes
    major_axis1 = pca1['eigenvectors'][:, 0]
    major_axis2 = pca2['eigenvectors'][:, 0]

    cos_angle = np.dot(major_axis1, major_axis2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # Take the acute angle
    if angle > 90:
        angle = 180 - angle

    print("\n" + "=" * 70)
    print(f"ORIENTATION DIFFERENCE: {angle:.2f}°")
    print("=" * 70 + "\n")

    return angle


# # Example usage:
# if __name__ == "__main__":
#     # Example 1: Create a tilted cylinder
#     print("Creating test point cloud (tilted cylinder)...")
#     mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=3.0)
#     mesh.compute_vertex_normals()
#
#     # Rotate it to make it interesting
#     R = mesh.get_rotation_matrix_from_xyz((np.pi / 6, np.pi / 4, 0))  # 30°, 45°, 0°
#     mesh.rotate(R, center=mesh.get_center())
#
#     pcd = mesh.sample_points_uniformly(number_of_points=5000)
#
#     # Compute PCA orientation
#     pca_result = compute_pca_orientation(pcd)
#
#     # Visualize with PCA axes
#     visualize_pca_axes(pcd, pca_result)
#
#     # Get orientation angle relative to Z-axis
#     angle = get_orientation_angle(pcd, reference_axis='z')
#
#     # Align to PCA axes
#     aligned_pcd = align_pointcloud_to_pca(pcd, pca_result)
#
#     # Visualize aligned point cloud
#     aligned_pcd.paint_uniform_color([0, 0.7, 1])
#     o3d.visualization.draw_geometries(
#         [aligned_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
#         window_name="Aligned Point Cloud",
#         width=1400,
#         height=900
#     )
#
#     # Example 2: Align major axis to Y-axis
#     print("\n--- Aligning Major Axis to Y-axis ---")
#     aligned_to_y = align_major_axis_to_axis(pcd, target_axis='y')
#     aligned_to_y.paint_uniform_color([1, 0.5, 0])
#     o3d.visualization.draw_geometries(
#         [aligned_to_y, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
#         window_name="Aligned to Y-axis",
#         width=1400,
#         height=900
#     )
#
#     # Example 3: Align PC2 (mid axis) to Z-axis
#     print("\n--- Aligning PC2 to Z-axis ---")
#     aligned_pc2_to_z = align_axis_to_axis(pcd, source_axis='pc1', target_axis='z')
#     aligned_pc2_to_z.paint_uniform_color([0.5, 1, 0.5])
#     o3d.visualization.draw_geometries(
#         [aligned_pc2_to_z, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
#         window_name="PC2 aligned to Z-axis",
#         width=1400,
#         height=900
#     )
#
#     # For your own point cloud:
#     """
#     # Load your point cloud
#     my_pcd = o3d.io.read_point_cloud("your_file.ply")
#
#     # Method 1: Align major axis (PC1) to Y-axis
#     aligned = align_major_axis_to_axis(my_pcd, target_axis='y')
#     o3d.io.write_point_cloud("aligned_to_y.ply", aligned)
#
#     # Method 2: Align specific PC axis to specific world axis
#     # Align PC2 to X-axis
#     aligned = align_axis_to_axis(my_pcd, source_axis='pc2', target_axis='x')
#
#     # Method 3: Full PCA alignment (PC1→X, PC2→Y, PC3→Z)
#     pca_result = compute_pca_orientation(my_pcd)
#     aligned = align_pointcloud_to_pca(my_pcd, pca_result)
#
#     # Visualize
#     o3d.visualization.draw_geometries([aligned])
#     """


import open3d as o3d
import numpy as np


def align_min_xyz(pcd1, pcd2, in_place=False):
    """
    Align two point clouds so they have the same minimum x, y, z values

    Args:
        pcd1: First point cloud
        pcd2: Second point cloud
        in_place: If True, modify original point clouds; if False, create copies

    Returns:
        tuple: (aligned_pcd1, aligned_pcd2)
    """
    # Create copies if not modifying in place
    if not in_place:
        pcd1 = o3d.geometry.PointCloud(pcd1)
        pcd2 = o3d.geometry.PointCloud(pcd2)

    # Get points
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # Get minimum values for each
    min1 = points1.min(axis=0)
    min2 = points2.min(axis=0)

    print(f"PCD1 min: [{min1[0]:.4f}, {min1[1]:.4f}, {min1[2]:.4f}]")
    print(f"PCD2 min: [{min2[0]:.4f}, {min2[1]:.4f}, {min2[2]:.4f}]")

    # Calculate the overall minimum
    overall_min = np.minimum(min1, min2)

    print(f"Target min: [{overall_min[0]:.4f}, {overall_min[1]:.4f}, {overall_min[2]:.4f}]")

    # Translate both to have the same minimum
    translation1 = overall_min - min1
    translation2 = overall_min - min2

    pcd1.translate(translation1)
    pcd2.translate(translation2)

    print(f"Translation 1: [{translation1[0]:.4f}, {translation1[1]:.4f}, {translation1[2]:.4f}]")
    print(f"Translation 2: [{translation2[0]:.4f}, {translation2[1]:.4f}, {translation2[2]:.4f}]")

    return pcd1, pcd2


def align_to_reference_min(pcd, reference_pcd):
    """
    Align a point cloud to match the minimum x, y, z of a reference point cloud

    Args:
        pcd: Point cloud to align
        reference_pcd: Reference point cloud

    Returns:
        Aligned point cloud
    """
    points = np.asarray(pcd.points)
    ref_points = np.asarray(reference_pcd.points)

    min_xyz = points.min(axis=0)
    ref_min_xyz = ref_points.min(axis=0)

    translation = ref_min_xyz - min_xyz
    pcd.translate(translation)

    print(f"Aligned to reference min: [{ref_min_xyz[0]:.4f}, {ref_min_xyz[1]:.4f}, {ref_min_xyz[2]:.4f}]")

    return pcd


def align_min_to_origin(pcd):
    """
    Translate point cloud so its minimum x, y, z is at origin (0, 0, 0)

    Args:
        pcd: Point cloud to align

    Returns:
        Aligned point cloud
    """
    points = np.asarray(pcd.points)
    min_xyz = points.min(axis=0)

    print(f"Original min: [{min_xyz[0]:.4f}, {min_xyz[1]:.4f}, {min_xyz[2]:.4f}]")

    pcd.translate(-min_xyz)

    print(f"Translated to origin: [0.0000, 0.0000, 0.0000]")

    return pcd


def get_bounding_box_info(pcd, name="Point Cloud"):
    """
    Print bounding box information
    """
    points = np.asarray(pcd.points)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    print(f"\n{name}:")
    print(f"  Min: [{min_xyz[0]:.4f}, {min_xyz[1]:.4f}, {min_xyz[2]:.4f}]")
    print(f"  Max: [{max_xyz[0]:.4f}, {max_xyz[1]:.4f}, {max_xyz[2]:.4f}]")
    print(f"  Size: [{max_xyz[0] - min_xyz[0]:.4f}, {max_xyz[1] - min_xyz[1]:.4f}, {max_xyz[2] - min_xyz[2]:.4f}]")


# import open3d as o3d
# import numpy as np


def visualize_with_transparency(pcd1, pcd2, color1=[1, 0, 0], color2=[0, 0, 1],
                                alpha1=0.5, alpha2=0.5):
    """
    Visualize two point clouds with transparency (requires custom renderer)
    Note: Open3D doesn't support true transparency, so we use color blending

    Args:
        pcd1: First point cloud
        pcd2: Second point cloud
        color1: Color for pcd1 [R, G, B]
        color2: Color for pcd2 [R, G, B]
        alpha1: Transparency for pcd1 (0-1, lower = more transparent)
        alpha2: Transparency for pcd2 (0-1, lower = more transparent)
    """
    # Make copies
    vis_pcd1 = o3d.geometry.PointCloud(pcd1)
    vis_pcd2 = o3d.geometry.PointCloud(pcd2)

    # Apply semi-transparent colors by making them lighter
    color1_transparent = [c * alpha1 + (1 - alpha1) for c in color1]
    color2_transparent = [c * alpha2 + (1 - alpha2) for c in color2]

    vis_pcd1.paint_uniform_color(color1_transparent)
    vis_pcd2.paint_uniform_color(color2_transparent)

    # Visualize
    o3d.visualization.draw_geometries(
        [vis_pcd1, vis_pcd2],
        window_name="Transparent Comparison",
        width=1400,
        height=900,
        point_show_normal=False
    )


def visualize_with_custom_renderer(pcd1, pcd2):
    """
    Use Open3D's advanced rendering for better transparency
    """
    # import open3d.visualization.gui as gui
    # import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Transparent Point Clouds", 1400, 900)
    widget3d = gui.SceneWidget()
    window.add_child(widget3d)

    widget3d.scene = rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([0.1, 0.1, 0.1, 1])

    # Material with transparency for pcd1
    mat1 = rendering.MaterialRecord()
    mat1.shader = "defaultUnlit"
    mat1.base_color = [1.0, 0.0, 0.0, 0.5]  # Red with 50% transparency
    mat1.point_size = 2.0

    # Material with transparency for pcd2
    mat2 = rendering.MaterialRecord()
    mat2.shader = "defaultUnlit"
    mat2.base_color = [0.0, 0.0, 1.0, 0.5]  # Blue with 50% transparency
    mat2.point_size = 2.0

    # Add point clouds
    widget3d.scene.add_geometry("pcd1", pcd1, mat1)
    widget3d.scene.add_geometry("pcd2", pcd2, mat2)

    # Setup camera
    bounds = pcd1.get_axis_aligned_bounding_box()
    widget3d.setup_camera(60, bounds, bounds.get_center())

    # Layout
    window.set_on_layout(lambda ctx: widget3d.frame == window.content_rect)

    app.run()


def visualize_difference_heatmap(pcd1, pcd2):
    """
    Show differences as a heatmap with semi-transparent overlay
    """
    # Compute distances
    dists = pcd1.compute_point_cloud_distance(pcd2)
    dists = np.asarray(dists)

    # Normalize distances
    max_dist = dists.max()
    normalized = dists / max_dist if max_dist > 0 else dists

    # Create color map (blue=similar, red=different)
    colors = np.zeros((len(dists), 3))
    colors[:, 0] = normalized  # Red channel
    colors[:, 2] = 1 - normalized  # Blue channel

    # Make it semi-transparent by brightening
    colors = colors * 0.7 + 0.3

    pcd1_colored = o3d.geometry.PointCloud(pcd1)
    pcd1_colored.colors = o3d.utility.Vector3dVector(colors)

    # Show reference as gray
    pcd2_gray = o3d.geometry.PointCloud(pcd2)
    pcd2_gray.paint_uniform_color([0.8, 0.8, 0.8])

    print(f"Max difference: {max_dist:.4f}")
    print(f"Mean difference: {dists.mean():.4f}")
    print("\nBlue = Similar, Red = Different")

    o3d.visualization.draw_geometries(
        [pcd1_colored, pcd2_gray],
        window_name="Difference Heatmap with Reference",
        width=1400,
        height=900
    )


def visualize_alternating(pcd1, pcd2, sample_rate=0.5):
    """
    Mix points from both clouds for a blended effect
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # Sample points from each
    n1 = int(len(points1) * sample_rate)
    n2 = int(len(points2) * sample_rate)

    idx1 = np.random.choice(len(points1), n1, replace=False)
    idx2 = np.random.choice(len(points2), n2, replace=False)

    # Create mixed point cloud
    mixed = o3d.geometry.PointCloud()
    mixed_points = np.vstack([points1[idx1], points2[idx2]])
    mixed.points = o3d.utility.Vector3dVector(mixed_points)

    # Color: first half red, second half blue
    colors = np.zeros((len(mixed_points), 3))
    colors[:n1] = [1, 0.3, 0.3]  # Light red
    colors[n1:] = [0.3, 0.3, 1]  # Light blue
    mixed.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [mixed],
        window_name="Mixed Point Clouds",
        width=1400,
        height=900
    )

### ----- compute oriented bounding and length, width, height -------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_aabb(points):
    """
    Compute axis-aligned bounding box
    Returns: min_corner, max_corner, dimensions (x, y, z), center
    """
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)

    # Dimensions in x, y, z directions
    dims = max_corner - min_corner

    # Center
    center = (min_corner + max_corner) / 2

    return min_corner, max_corner, dims, center


def get_aabb_corners(min_corner, max_corner):
    """
    Get the 8 corners of the axis-aligned bounding box
    """
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner

    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])

    return corners


def plot_aabb(ax, corners, color, label):
    """
    Plot the axis-aligned bounding box edges
    """
    # Define the 12 edges of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
    ]

    for edge in edges:
        pts = corners[edge]
        ax.plot3D(*pts.T, color=color, linewidth=2, alpha=0.8)

    # Add label to one edge
    ax.plot3D([], [], [], color=color, linewidth=2, label=label)


### -------------------------------------------
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PointCloudSlicer:
    """
    Robust point cloud slicing and girth calculation.
    Handles noise, irregular sampling, and minor differences between clouds.
    """

    def __init__(self, tolerance=0.01):
        """
        Args:
            tolerance: Distance threshold for points near the cutting plane
        """
        self.tolerance = tolerance

    def _to_numpy(self, points):
        """
        Convert various point cloud formats to numpy array.

        Args:
            points: Can be numpy array, Open3D PointCloud, or list

        Returns:
            Nx3 numpy array
        """
        # Check if it's an Open3D PointCloud
        if hasattr(points, 'points'):
            # Open3D PointCloud object
            return np.asarray(points.points, dtype=np.float64)
        elif isinstance(points, np.ndarray):
            return points.astype(np.float64)
        else:
            # Try to convert to numpy array
            return np.asarray(points, dtype=np.float64)

    def slice_with_plane(self, points, plane_origin, plane_normal):
        """
        Extract points near a plane and project them onto it.

        Args:
            points: Nx3 array of point coordinates or Open3D PointCloud object
            plane_origin: 3D point on the plane
            plane_normal: 3D normal vector of the plane

        Returns:
            slice_points_3d: Points near the plane (3D)
            slice_points_2d: Points projected onto plane (2D coordinates)
            indices: Original indices of sliced points
        """
        # Convert Open3D PointCloud to numpy array if needed
        points = self._to_numpy(points)

        # Ensure plane_origin and plane_normal are numpy arrays
        plane_origin = np.asarray(plane_origin, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)

        # Normalize the plane normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Calculate signed distance from each point to the plane
        distances = np.dot(points - plane_origin, plane_normal)

        # Select points within tolerance of the plane
        mask = np.abs(distances) <= self.tolerance
        slice_points_3d = points[mask]
        indices = np.where(mask)[0]

        if len(slice_points_3d) < 3:
            raise ValueError(f"Insufficient points in slice: {len(slice_points_3d)}")

        # Project points onto the plane
        projected = slice_points_3d - np.outer(distances[mask], plane_normal)

        # Create 2D coordinate system on the plane
        slice_points_2d = self._project_to_2d(projected, plane_origin, plane_normal)

        return slice_points_3d, slice_points_2d, indices

    def _project_to_2d(self, points_3d, plane_origin, plane_normal):
        """
        Project 3D points onto a 2D coordinate system in the plane.
        """
        # Create orthonormal basis for the plane
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Choose arbitrary perpendicular vector
        if abs(plane_normal[0]) < 0.9:
            u = np.cross(plane_normal, [1, 0, 0])
        else:
            u = np.cross(plane_normal, [0, 1, 0])
        u = u / np.linalg.norm(u)

        # Second basis vector
        v = np.cross(plane_normal, u)
        v = v / np.linalg.norm(v)

        # Project onto 2D basis
        relative = points_3d - plane_origin
        points_2d = np.column_stack([
            np.dot(relative, u),
            np.dot(relative, v)
        ])

        return points_2d

    def calculate_girth(self, points_2d, method='convex_hull', smooth=True, num_sectors=360):
        """
        Calculate girth (perimeter) from 2D slice points.

        Args:
            points_2d: Nx2 array of 2D coordinates
            method: 'convex_hull', 'dense_boundary', 'alpha_shape', or 'fitted_curve'
            smooth: Whether to smooth the boundary
            num_sectors: Number of angular sectors for dense_boundary method

        Returns:
            girth: Perimeter length
            boundary_points: Points forming the boundary
        """
        if len(points_2d) < 3:
            raise ValueError("Need at least 3 points to calculate girth")

        if method == 'convex_hull':
            return self._girth_convex_hull(points_2d)
        elif method == 'dense_boundary':
            return self._girth_dense_boundary(points_2d, num_sectors)
        elif method == 'alpha_shape':
            return self._girth_alpha_shape(points_2d)
        elif method == 'fitted_curve':
            return self._girth_fitted_curve(points_2d, smooth)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _girth_convex_hull(self, points_2d):
        """Calculate girth using convex hull (fastest, works for convex shapes)."""
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        # Calculate perimeter
        girth = 0
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            girth += np.linalg.norm(p2 - p1)

        return girth, hull_points

    def _girth_dense_boundary(self, points_2d, num_sectors=360):
        """
        Calculate girth by finding boundary points in angular sectors.
        Better matches data points than convex hull.

        Args:
            points_2d: Nx2 array of 2D coordinates
            num_sectors: Number of angular sectors to divide the circle
        """
        # Find center
        center = np.mean(points_2d, axis=0)

        # Calculate angles and distances from center
        relative = points_2d - center
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        distances = np.linalg.norm(relative, axis=1)

        # Divide into angular sectors and find furthest point in each
        sector_angles = np.linspace(-np.pi, np.pi, num_sectors + 1)
        boundary_points = []

        for i in range(num_sectors):
            angle_min = sector_angles[i]
            angle_max = sector_angles[i + 1]

            # Find points in this sector
            if i == num_sectors - 1:  # Last sector includes upper boundary
                mask = (angles >= angle_min) & (angles <= angle_max)
            else:
                mask = (angles >= angle_min) & (angles < angle_max)

            if np.any(mask):
                # Get the furthest point in this sector
                sector_points = points_2d[mask]
                sector_distances = distances[mask]
                furthest_idx = np.argmax(sector_distances)
                boundary_points.append(sector_points[furthest_idx])

        if len(boundary_points) < 3:
            # Fallback to convex hull if not enough sectors have points
            return self._girth_convex_hull(points_2d)

        boundary_points = np.array(boundary_points)

        # Calculate perimeter
        girth = 0
        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]
            girth += np.linalg.norm(p2 - p1)

        return girth, boundary_points

    def _girth_alpha_shape(self, points_2d, alpha=None):
        """
        Calculate girth using alpha shape (better for concave boundaries).
        Simplified implementation using adaptive radius.
        """
        if alpha is None:
            # Estimate alpha from point density
            dists = distance_matrix(points_2d, points_2d)
            np.fill_diagonal(dists, np.inf)
            avg_nearest = np.mean(np.min(dists, axis=1))
            alpha = 2.0 / avg_nearest

        # For simplicity, use convex hull with outlier removal
        center = np.mean(points_2d, axis=0)
        radii = np.linalg.norm(points_2d - center, axis=1)
        median_radius = np.median(radii)

        # Keep points within reasonable distance
        mask = radii <= median_radius * 1.5
        filtered = points_2d[mask]

        hull = ConvexHull(filtered)
        hull_points = filtered[hull.vertices]

        girth = 0
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            girth += np.linalg.norm(p2 - p1)

        return girth, hull_points

    def _girth_fitted_curve(self, points_2d, smooth=True):
        """
        Fit a smooth curve through points (best for noisy data).
        """
        # Use PCA to order points roughly by angle
        center = np.mean(points_2d, axis=0)
        centered = points_2d - center
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        sorted_idx = np.argsort(angles)
        sorted_points = points_2d[sorted_idx]

        if smooth and len(sorted_points) > 10:
            # Fit a smooth spline
            tck, u = splprep([sorted_points[:, 0], sorted_points[:, 1]],
                             s=len(sorted_points) * 0.01, per=True)
            u_new = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_new, tck)
            curve_points = np.column_stack([x_new, y_new])
        else:
            curve_points = sorted_points

        # Calculate perimeter
        girth = 0
        for i in range(len(curve_points)):
            p1 = curve_points[i]
            p2 = curve_points[(i + 1) % len(curve_points)]
            girth += np.linalg.norm(p2 - p1)

        return girth, curve_points

    def compare_girths(self, pc1, pc2, plane_origin, plane_normal, method='convex_hull'):
        """
        Compare girths of two point clouds at the same cutting plane.

        Args:
            pc1, pc2: Point clouds (numpy arrays or Open3D PointCloud objects)
            plane_origin: Point on the cutting plane
            plane_normal: Normal vector of the cutting plane
            method: Girth calculation method

        Returns:
            dict with girth values and difference statistics
        """
        # Convert inputs to numpy arrays
        pc1 = self._to_numpy(pc1)
        pc2 = self._to_numpy(pc2)
        plane_origin = np.asarray(plane_origin, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)

        # Slice both point clouds
        _, slice1_2d, _ = self.slice_with_plane(pc1, plane_origin, plane_normal)
        _, slice2_2d, _ = self.slice_with_plane(pc2, plane_origin, plane_normal)

        # Calculate girths
        girth1, boundary1 = self.calculate_girth(slice1_2d, method)
        girth2, boundary2 = self.calculate_girth(slice2_2d, method)

        return {
            'girth1': girth1,
            'girth2': girth2,
            'difference': girth2 - girth1,
            'percent_change': 100 * (girth2 - girth1) / girth1,
            'boundary1': boundary1,
            'boundary2': boundary2,
            'slice1_2d': slice1_2d,
            'slice2_2d': slice2_2d
        }

    def visualize_slice_overlay(self, slice1_2d, boundary1, slice2_2d, boundary2,
                                girth1, girth2, title="Overlaid Slice Comparison", ax=None):
        """Visualize two slices overlaid on the same plot with different colors."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()

        # Plot first point cloud in blue
        ax.scatter(slice1_2d[:, 0], slice1_2d[:, 1],
                   c='lightblue', s=20, alpha=0.5, label='PC1 slice points')
        ax.plot(np.append(boundary1[:, 0], boundary1[0, 0]),
                np.append(boundary1[:, 1], boundary1[0, 1]),
                'b-', linewidth=2, label=f'PC1 boundary (G={girth1:.3f})')

        # Plot second point cloud in orange
        ax.scatter(slice2_2d[:, 0], slice2_2d[:, 1],
                   c='orange', s=20, alpha=0.5, label='PC2 slice points')
        ax.plot(np.append(boundary2[:, 0], boundary2[0, 0]),
                np.append(boundary2[:, 1], boundary2[0, 1]),
                'r-', linewidth=2, label=f'PC2 boundary (G={girth2:.3f})')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return ax.figure
        """Visualize the 2D slice and boundary."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()

        ax.scatter(slice_2d[:, 0], slice_2d[:, 1],
                   c='lightblue', s=20, alpha=0.6, label='Slice points')
        ax.plot(np.append(boundary[:, 0], boundary[0, 0]),
                np.append(boundary[:, 1], boundary[0, 1]),
                'r-', linewidth=2, label='Boundary')
        ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return ax.figure

    def visualize_3d_with_plane(self, points, plane_origin, plane_normal,
                                slice_indices=None, title="Point Cloud with Cutting Plane",
                                show_plane_mesh=True, plane_size=2.0):
        """
        Visualize point cloud with cutting plane and highlighted slice points in 3D.

        Args:
            points: Nx3 array or Open3D PointCloud
            plane_origin: 3D point on plane
            plane_normal: 3D normal vector
            slice_indices: Indices of points in the slice (if None, will compute)
            title: Plot title
            show_plane_mesh: Whether to show plane as a mesh
            plane_size: Size of the plane visualization
        """
        points = self._to_numpy(points)
        plane_origin = np.asarray(plane_origin, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Get slice indices if not provided
        if slice_indices is None:
            distances = np.dot(points - plane_origin, plane_normal)
            slice_indices = np.where(np.abs(distances) <= self.tolerance)[0]

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot non-slice points in gray
        non_slice_mask = np.ones(len(points), dtype=bool)
        non_slice_mask[slice_indices] = False
        if np.any(non_slice_mask):
            ax.scatter(points[non_slice_mask, 0],
                       points[non_slice_mask, 1],
                       points[non_slice_mask, 2],
                       c='lightgray', s=1, alpha=0.3, label='Point cloud')

        # Plot slice points in red
        if len(slice_indices) > 0:
            ax.scatter(points[slice_indices, 0],
                       points[slice_indices, 1],
                       points[slice_indices, 2],
                       c='red', s=20, alpha=0.8, label='Slice points', edgecolors='darkred')

        # Create plane mesh for visualization
        if show_plane_mesh:
            # Create orthonormal basis for the plane
            if abs(plane_normal[0]) < 0.9:
                u = np.cross(plane_normal, [1, 0, 0])
            else:
                u = np.cross(plane_normal, [0, 1, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(plane_normal, u)

            # Create plane grid
            u_range = np.linspace(-plane_size, plane_size, 10)
            v_range = np.linspace(-plane_size, plane_size, 10)
            plane_points = []
            for u_val in u_range:
                for v_val in v_range:
                    point = plane_origin + u_val * u + v_val * v
                    plane_points.append(point)
            plane_points = np.array(plane_points)

            # Plot plane as surface
            U, V = np.meshgrid(u_range, v_range)
            X = plane_origin[0] + U * u[0] + V * v[0]
            Y = plane_origin[1] + U * u[1] + V * v[1]
            Z = plane_origin[2] + U * u[2] + V * v[2]
            ax.plot_surface(X, Y, Z, alpha=0.3, color='blue', label='Cutting plane')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()

        # Equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    def visualize_comparison_3d(self, pc1, pc2, plane_origin, plane_normal,
                                slice1_indices=None, slice2_indices=None):
        """
        Create side-by-side 3D visualization of both point clouds with cutting plane.
        """
        pc1 = self._to_numpy(pc1)
        pc2 = self._to_numpy(pc2)

        # Get slice indices if not provided
        if slice1_indices is None:
            distances = np.dot(pc1 - plane_origin, plane_normal / np.linalg.norm(plane_normal))
            slice1_indices = np.where(np.abs(distances) <= self.tolerance)[0]
        if slice2_indices is None:
            distances = np.dot(pc2 - plane_origin, plane_normal / np.linalg.norm(plane_normal))
            slice2_indices = np.where(np.abs(distances) <= self.tolerance)[0]

        fig = plt.figure(figsize=(20, 9))

        # Plot first point cloud
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_cloud_with_slice(ax1, pc1, plane_origin, plane_normal,
                                    slice1_indices, "Point Cloud 1")

        # Plot second point cloud
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_cloud_with_slice(ax2, pc2, plane_origin, plane_normal,
                                    slice2_indices, "Point Cloud 2")

        plt.tight_layout()
        return fig

    def _plot_cloud_with_slice(self, ax, points, plane_origin, plane_normal,
                               slice_indices, title):
        """Helper to plot a single cloud with slice on given axes."""
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Plot non-slice points
        non_slice_mask = np.ones(len(points), dtype=bool)
        non_slice_mask[slice_indices] = False
        if np.any(non_slice_mask):
            ax.scatter(points[non_slice_mask, 0],
                       points[non_slice_mask, 1],
                       points[non_slice_mask, 2],
                       c='lightgray', s=1, alpha=0.3, label='Point cloud')

        # Plot slice points
        if len(slice_indices) > 0:
            ax.scatter(points[slice_indices, 0],
                       points[slice_indices, 1],
                       points[slice_indices, 2],
                       c='red', s=20, alpha=0.8, label='Slice points', edgecolors='darkred')

        # Plot plane
        if abs(plane_normal[0]) < 0.9:
            u = np.cross(plane_normal, [1, 0, 0])
        else:
            u = np.cross(plane_normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(plane_normal, u)

        plane_size = np.max([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]) * 0.6

        u_range = np.linspace(-plane_size / 2, plane_size / 2, 10)
        v_range = np.linspace(-plane_size / 2, plane_size / 2, 10)
        U, V = np.meshgrid(u_range, v_range)
        X = plane_origin[0] + U * u[0] + V * v[0]
        Y = plane_origin[1] + U * u[1] + V * v[1]
        Z = plane_origin[2] + U * u[2] + V * v[2]
        ax.plot_surface(X, Y, Z, alpha=0.3, color='blue')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()

        # Equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def visualize_slice(self, slice_2d, boundary, title="Point Cloud Slice", ax=None):
        """Visualize the 2D slice and boundary."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()

        ax.scatter(slice_2d[:, 0], slice_2d[:, 1],
                   c='lightblue', s=20, alpha=0.6, label='Slice points')
        ax.plot(np.append(boundary[:, 0], boundary[0, 0]),
                np.append(boundary[:, 1], boundary[0, 1]),
                'r-', linewidth=2, label='Boundary')
        ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return ax.figure


import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


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


import numpy as np


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


import numpy as np
from scipy.interpolate import interp1d


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


import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def estimate_eps(pcd, k=10):
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


def get_largest_cluster_auto(pcd, min_points=10):
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


import numpy as np
import open3d as o3d


def segment_plane_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
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


import numpy as np
import open3d as o3d


def extract_points_by_y_range(pcd, y_min, y_max):
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


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def remove_bridges_complete(pcd, radius=2.0, density_percentile=30,
                            min_cluster_size=100, visualize=True):
    """
    Complete pipeline to remove narrow bridges.
    """
    print("=" * 60)
    print("NARROW CONNECTION REMOVAL PIPELINE")
    print("=" * 60)

    original_count = len(pcd.points)
    print(f"Input: {original_count} points")

    # Step 1: Compute local density
    print(f"\nStep 1: Computing local density (radius={radius})")
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    densities = []
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        densities.append(k)

    densities = np.array(densities)

    # Visualize density distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.hist(densities, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.percentile(densities, density_percentile),
                    color='red', linestyle='--', linewidth=2,
                    label=f'{density_percentile}th percentile')
        plt.xlabel('Local Density (neighbor count)')
        plt.ylabel('Frequency')
        plt.title('Point Density Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print(f"  Density range: [{densities.min()}, {densities.max()}]")
    print(f"  Mean density: {densities.mean():.1f}")

    # Step 2: Filter low-density points
    threshold = np.percentile(densities, density_percentile)
    print(f"\nStep 2: Filtering points with density < {threshold:.1f}")

    high_density_mask = densities > threshold
    filtered_indices = np.where(high_density_mask)[0]
    filtered_pcd = pcd.select_by_index(filtered_indices)

    print(f"  Removed: {original_count - len(filtered_pcd.points)} points")
    print(f"  Remaining: {len(filtered_pcd.points)} points")

    # Step 3: Cluster remaining points
    print(f"\nStep 3: Clustering (eps={radius})")
    labels = np.array(filtered_pcd.cluster_dbscan(eps=radius, min_points=10))

    unique_labels = np.unique(labels[labels >= 0])
    print(f"  Found {len(unique_labels)} clusters")

    # Step 4: Keep large clusters
    print(f"\nStep 4: Filtering clusters (min size={min_cluster_size})")
    valid_clusters = []
    cluster_sizes = []

    for label in unique_labels:
        size = np.sum(labels == label)
        cluster_sizes.append(size)
        if size >= min_cluster_size:
            valid_clusters.append(label)
            print(f"  Cluster {label}: {size} points - KEPT")
        else:
            print(f"  Cluster {label}: {size} points - REMOVED")

    if len(valid_clusters) == 0:
        print("\nWarning: No clusters meet size threshold!")
        return filtered_pcd

    # Extract final result
    valid_mask = np.isin(labels, valid_clusters)
    cleaned_pcd = filtered_pcd.select_by_index(np.where(valid_mask)[0])

    print(f"\n{'=' * 60}")
    print("RESULTS:")
    print(f"  Original points: {original_count}")
    print(f"  Final points: {len(cleaned_pcd.points)}")
    print(
        f"  Removed: {original_count - len(cleaned_pcd.points)} ({(1 - len(cleaned_pcd.points) / original_count) * 100:.1f}%)")
    print(f"  Clusters kept: {len(valid_clusters)}")
    if len(valid_clusters) > 0:
        print(f"  Largest cluster: {max(cluster_sizes)} points")
    print("=" * 60)

    # Visualize if requested
    if visualize:
        # Color different clusters
        if len(valid_clusters) > 1:
            colors = plt.get_cmap("tab10")(np.arange(len(valid_clusters)) / len(valid_clusters))
            colored_pcd = o3d.geometry.PointCloud(cleaned_pcd)

            point_colors = np.zeros((len(cleaned_pcd.points), 3))
            labels_final = np.array(cleaned_pcd.cluster_dbscan(eps=radius, min_points=10))

            for i, label in enumerate(valid_clusters):
                mask = labels_final == label
                point_colors[mask] = colors[i, :3]

            colored_pcd.colors = o3d.utility.Vector3dVector(point_colors)
        else:
            colored_pcd = cleaned_pcd
            colored_pcd.paint_uniform_color([0, 0.7, 0])

        o3d.visualization.draw_geometries([colored_pcd],
                                          window_name="Cleaned Point Cloud")

    return cleaned_pcd

import gc
import matplotlib.pyplot as plt

def clean_up():
    # Explicit cleanup
    gc.collect()
    print("✓ Memory cleanup completed")

    # Delete Large Variables
    # del big_array, huge_dict  # remove references
    # try:
    #     plt.close('all')
    #     print("✓ close all plots")
    # except:
    #     pass
def close_all_plots():
    try:
        plt.close('all')
        print("✓ close all plots")
    except:
        pass


### ------------- end of function region --------------------------


### ---------------------------------------------------------------



### ------------------------------------------------
### ----------- main code ---------------------
# *********** load mesh from file ***********
#foot_id = "right"
foot_id = "left"

import os
from pathlib import Path

# Dayong Starts
import copy
from helper import *

cur_dir = Path(__file__).resolve().parent
dayong_dir = cur_dir.parent

# Scanner's Scans
stationary_case = '/Jian_Gong FullWeight3_550013_000026_R'
stationary_case_path = os.path.join(dayong_dir, "scans", "STLs", "Scanner")

# iPhone's Scans
# stationary_case = '/right_foot_mesh_j'
# stationary_case_path = os.path.join(dayong_dir, "scans", "STLs", "iPhoneScans")

mobile_case_path = os.path.join(dayong_dir, "scans", "STLs")

# insole path
insole_path = os.path.join(dayong_dir, "scans", "STLs", "iPhoneScans", "hd-0215.stl")

# load stationary mesh
if foot_id == 'left':
    stl_foot_id = 'L'
else:
    stl_foot_id = 'R'
# stationary_scan_path = file_path_base+"right_foot_mesh.stl"
# stationary_scan_path= '/Users/nic_gong/Documents/StationaryScanData/StationaryScanValidationData/Jian_Gong_Fullweight4_550013_000027/Jian_Gong Fullweight4_550013_000027_L.stl'
stationary_scan_path = stationary_case_path+stationary_case +'.stl' ### +'_'+stl_foot_id
print(stationary_scan_path)
stationary_o3d_mesh = o3d.io.read_triangle_mesh(stationary_scan_path)
stationary_o3d_mesh = normalize_mesh_to_mm(stationary_o3d_mesh)

# insole mesh
insole_mesh = o3d.io.read_triangle_mesh(insole_path)
insole_mesh = normalize_mesh_to_mm(insole_mesh)

### load the file
# load mobile scanned mesh
# scan_mesh_path = mobile_case_path + '_'+stl_foot_id+".stl"
# scan_mesh_path = mobile_case_path + foot_id +"_foot_mesh_refine.stl"
scan_mesh_path = mobile_case_path + "/" + foot_id +"_foot_mesh.stl"

scan_o3d_mesh = o3d.io.read_triangle_mesh(scan_mesh_path)
scan_o3d_mesh.scale(1000.0, center=scan_o3d_mesh.get_center())


# Align point clouds first if needed
# pcd2.transform(transformation_matrix)
pcd1 = stationary_o3d_mesh
pcd2 = scan_o3d_mesh
insole_copy = copy.deepcopy(insole_mesh)
insole_copy.compute_vertex_normals()

n_points = 10000 * 18     ### number of points to analyze
# Convert meshes to point clouds
if isinstance(pcd1, o3d.geometry.TriangleMesh):
    pcd1_points = pcd1.sample_points_uniformly(number_of_points=n_points)
else:
    pcd1_points = pcd1

if isinstance(pcd2, o3d.geometry.TriangleMesh):
    pcd2_points = pcd2.sample_points_uniformly(number_of_points=n_points)
else:
    pcd2_points = pcd2

# insole pcd
if isinstance(insole_copy, o3d.geometry.TriangleMesh):
    insole_points = insole_copy.sample_points_uniformly(number_of_points=n_points)
else:
    insole_points = insole_copy

# visualize the results
# show the results
o3d.visualization.draw_geometries([
    pcd1_points.paint_uniform_color([1, 0, 0]),
    #pcd2_points.paint_uniform_color([0, 1, 0]),
    ], window_name="check read STL files Stationary and phone")

# view the insole
o3d.visualization.draw_geometries([
    insole_points.paint_uniform_color([1, 0, 0]),
    ], window_name="Visualize the pcd of the insole")

### ------ cut the foot to get the orientation right
# Method 1: Filter from min Z with specific distance
print("\n--- Method 1: Filter from min Z + distance ---")
distance = 20.0  # Keep 2.0 units from the bottom
filtered_pcd1 = filter_from_min_z_with_distance(pcd1_points, distance_from_min=distance)
filtered_pcd2 = filter_from_min_z_with_distance(pcd2_points, distance_from_min=distance)

distance = 100.0  # Keep 2.0 units from the bottom
cut_pcd1 = filter_from_min_z_with_distance(pcd1_points, distance_from_min=distance)
cut_pcd2 = filter_from_min_z_with_distance(pcd2_points, distance_from_min=distance)

# # Visualize
# print("------pcd1------")
# visualize_filtered_comparison(pcd1_points, filtered_pcd1)
# print("------pcd2------")
# visualize_filtered_comparison(pcd2_points, filtered_pcd2)

### ------------ rotate the orientation ----------------------
# get rotation right for pcd1
pca_result1 = compute_pca_orientation(filtered_pcd1)
print("\n--- Aligning Major Axis to Y-axis ---PCD1")
aligned_to_y_bot1, _ = align_major_axis_to_axis(filtered_pcd1, target_axis='y', pca_result=pca_result1)
aligned_to_y1, _ = align_major_axis_to_axis(cut_pcd1, target_axis='y', pca_result=pca_result1)

# get rotation right for the insole
pca_insole = compute_pca_orientation(insole_points)
print("\n--- Aligning Major Axis to Y-axis ---PCD Insole")
aligned_to_y_insole, insole_align_T = align_major_axis_to_axis(insole_points, target_axis='y', pca_result=pca_insole)

aligned_to_y_bot1.paint_uniform_color([0, 0.5, 0])
aligned_to_y1.paint_uniform_color([1, 0.5, 0])

# paint the insole
aligned_to_y_insole.paint_uniform_color([1, 0.5, 0])

# Flip point cloud by 180 degrees around given axis (x/y/z)
def flip_pointcloud_180(pcd, axis='x'):
    if axis == 'x':
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
    elif axis == 'y':
        R = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
        ])
    elif axis == 'z':
        R = np.array([
            [-1, 0, 0],
            [ 0,-1, 0],
            [ 0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    c = pcd.get_center()  # 注意：这是“当前 pcd”中心
    t_flip = get_translation_matrix_from_rotation(R, c)
    pcd.transform(t_flip)
    return pcd, t_flip

aligned_to_y_insole, insole_flip_T = flip_pointcloud_180(aligned_to_y_insole, axis='y')

# o3d.visualization.draw_geometries(
#             [aligned_to_y1, aligned_to_y_bot1, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
#             window_name="Aligned to Y-axis: Stationary",
#             width=1400,
#             height=900
#         )

o3d.visualization.draw_geometries(
            [aligned_to_y_bot1, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
            window_name="Aligned to Y-axis: Stationary",
            width=1400,
            height=900
        )

# view the insole
o3d.visualization.draw_geometries(
            [aligned_to_y_insole, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
            window_name="Aligned to Y-axis: insole",
            width=1400,
            height=900
        )

# insole - foot alignment
# aligned_to_y_insole, aligned_to_y_bot1 = align_min_xyz(aligned_to_y_insole, aligned_to_y_bot1)
aligned_to_y_insole, tx = align_min_x(aligned_to_y_insole, aligned_to_y_bot1)
aligned_to_y_insole, ty = align_min_y(aligned_to_y_insole, aligned_to_y_bot1)
aligned_to_y_insole, tz = align_z(aligned_to_y_insole, aligned_to_y_bot1, 13)

o3d.visualization.draw_geometries(
            [aligned_to_y_insole, aligned_to_y_bot1, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
            window_name="Aligned to Y-axis: insole",
            width=1400,
            height=900
        )

### project into the XY plane
xyz_points = aligned_to_y_bot1.points
xy_points = np.asarray(xyz_points)[:, 0:2]

#import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(xy_points[:, 0], xy_points[:, 1], s=1, c='blue', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')  # Equal aspect ratio
plt.title('XY Point Cloud')
plt.show()


###  ----- find the boundary -----------------
# from scipy.spatial import ConvexHull
#
# hull = ConvexHull(xy_points)
# boundary = xy_points[hull.vertices]
#
# plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='blue', alpha=0.3)
# plt.plot(boundary[:, 0], boundary[:, 1], 'r-', linewidth=2)
# plt.plot([boundary[-1, 0], boundary[0, 0]],
#          [boundary[-1, 1], boundary[0, 1]], 'r-', linewidth=2)  # Close the loop
# plt.axis('equal')
# plt.show()

# # Usage
# alpha = 5.0  # Adjust this value based on your data
# boundary_pts, _ = alpha_shape(xy_points, alpha)
#
# plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='blue', alpha=0.3)
# plt.scatter(boundary_pts[:, 0], boundary_pts[:, 1], s=2, c='red')
# plt.axis('equal')
# plt.show()

# Usage
alpha = 5.0  # Adjust based on your data scale
boundary = alpha_shape_boundary(xy_points, alpha)

# Plot
plt.figure(figsize=(12, 10))
plt.scatter(xy_points[:, 0], xy_points[:, 1], s=0.5, c='lightblue', alpha=0.3, label='All points')
plt.plot(boundary[:, 0], boundary[:, 1], 'r', linewidth=2, label='Boundary') ### r-
plt.plot([boundary[-1, 0], boundary[0, 0]],
         [boundary[-1, 1], boundary[0, 1]], 'r-', linewidth=2)  # Close the loop
plt.axis('equal')
plt.legend()
plt.title('Continuous Alpha Shape Boundary')
plt.show()

print(f"Boundary has {len(boundary)} points")


### ------------- line equal distance --------------------
# Usage
equal_boundary = resample_boundary_equal_distance(boundary, num_points=1000)

# Verify spacing
distances = np.sqrt(np.sum(np.diff(equal_boundary, axis=0) ** 2, axis=1))
print(f"Mean distance: {distances.mean():.6f}")
print(f"Std distance: {distances.std():.6f}")
print(f"Min distance: {distances.min():.6f}")
print(f"Max distance: {distances.max():.6f}")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
plt.plot(boundary[:, 0], boundary[:, 1], 'b-', alpha=0.3, label='Original')
plt.scatter(equal_boundary[:, 0], equal_boundary[:, 1], s=2, c='red', label='Equal spacing')
plt.axis('equal')
plt.legend()
plt.title('Resampled Boundary with Equal Spacing')
plt.show()

### breakpoint()


### ------------- shrink the boundary line ----------------
# Usage
shrunken = shrink_boundary(equal_boundary, shrink_ratio=0.01)  # 1% or 5% shrink

# Plot comparison
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

### ------------- prepare the cutting plane ----------------
# Get minimum bound of the point cloud
min_bound = aligned_to_y_bot1.get_min_bound()
min_z = min_bound[2]  # Z is the third component
print(f"Minimum Z value: {min_z}")

max_z = min_z + 30

#import numpy as np

# # Add constant Z value
# z_value = min_z  # or any constant value
# shrunken_3d = np.column_stack([shrunken, np.full(len(shrunken), z_value)])
#
# print(f"Original shape: {shrunken.shape}")      # (N, 2)
# print(f"3D shape: {shrunken_3d.shape}")         # (N, 3)
#
# # import open3d as o3d
# # import numpy as np

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

#### ------------------------- main code region ---------------------

# Assign different colors to each point cloud
pcd1 = aligned_to_y_bot1
pcd2 = layers_pcd

pcd11_vis = pcd1.paint_uniform_color([1, 0, 0])  # Red
pcd22_vis = pcd2.paint_uniform_color([0, 0, 1])  # Blue

# Visualize original clouds with box
o3d.visualization.draw_geometries([pcd11_vis, pcd22_vis],
                                      window_name="Original - Red: pcd1, Blue: pcd2")

print("\nSurface Processing...")
### cut the foot by the artificial knife
remain_pcd1 = call_surface_cut(pcd1, pcd2)  ### pcd1: target; pcd2: knift. Point Cloud
#remain_pcd2 = call_surface_cut(pcd2, pcd1)  ### pcd1: target; pcd2: knift. Point Cloud

# Get bounding box bounds
min_bound = pcd1.get_min_bound()
max_bound = pcd1.get_max_bound()

min_y = min_bound[1]  # Y is index 1
max_y = max_bound[1]

# Usage
y_min = min_y + (max_y-min_y)*0.01  # Your minimum Y value
y_max = min_y + (max_y-min_y)*0.8  # Your maximum Y value

cleaned_pcd = extract_points_by_y_range(remain_pcd1, y_min, y_max)
# Usage
#largest_piece = get_largest_cluster_auto(remain_pcd1, min_points=50)
largest_piece = get_largest_cluster_auto(cleaned_pcd, min_points=50)

pcd1_vis = remain_pcd1.paint_uniform_color([1, 0, 0])  # Red
pcd2_vis = largest_piece.paint_uniform_color([0, 0, 1])  # Blue
# pcd3_vis = largest_cluster_M1.paint_uniform_color([1, 0, 1])  # Magenta
# pcd4_vis = largest_cluster_M3.paint_uniform_color([0, 1, 0])  # Green

# Visualize original clouds with box
o3d.visualization.draw_geometries([largest_piece],
                                      window_name="Processed - Blue")

# o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis],
#                                       window_name="Processed - Red: pcd1, Blue: pcd2")

# o3d.visualization.draw_geometries([pcd1_vis],
#                                       window_name="Processed - Red: pcd1, Blue: pcd2")

# Usage
plane_pcd, non_plane_pcd, plane_model = segment_plane_ransac(
    largest_piece,
    distance_threshold=1.5,  # Adjust based on your data
    ransac_n=3,
    num_iterations=200
)

arch_raw = get_largest_cluster_auto(non_plane_pcd, min_points=50)
pcd4_vis = arch_raw.paint_uniform_color([0, 1, 0])  # Green

###   cut the arch region
# Get bounding box bounds
min_bound = pcd1.get_min_bound()
max_bound = pcd1.get_max_bound()

min_y = min_bound[1]  # Y is index 1
max_y = max_bound[1]

# Usage
y_min = min_y + (max_y-min_y)*0.2  # Your minimum Y value
y_max = min_y + (max_y-min_y)*0.7  # Your maximum Y value

filtered_pcd = extract_points_by_y_range(arch_raw, y_min, y_max)
filtered_pcd = get_largest_cluster_auto(filtered_pcd, min_points=50)
pcd5_vis = filtered_pcd.paint_uniform_color([1, 0, 1])  # Magenta
o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

# visualize the foot and the insole
o3d.visualization.draw_geometries([filtered_pcd, aligned_to_y_insole], window_name="arch - insole visualization")
# arch = crop_pcd_to_aabb_strict(filtered_pcd, aligned_to_y_insole)
# o3d.visualization.draw_geometries([arch, aligned_to_y_insole], window_name="arch (cropped) - insole visualization")

arch_mesh = pcd_to_mesh_bpa(filtered_pcd)
# o3d.visualization.draw_geometries([arch_mesh], window_name="arch (cropped) - insole visualization", mesh_show_back_face = True)

# ----------------- Apply insole transforms to the ORIGINAL insole mesh -----------------
# We have tracked every step as a 4x4 transform T (pcd stays the same; mesh must be synced).
# Order matters: the later transform in code is applied later, so the total is:
#   T_total = T_last @ ... @ T_first

# If your align_min_* helpers return T (as 4x4 matrices), then tx/ty/tz are those T's.
# Total insole transform (PCA-align -> flip -> min-x -> min-y -> z-align)
t_insole = tz @ ty @ tx @ insole_flip_T @ insole_align_T

# Apply to a COPY of the original insole mesh (keep original mesh intact)
insole_mesh_aligned = copy.deepcopy(insole_mesh)
insole_mesh_aligned.compute_vertex_normals()
insole_mesh_aligned.transform(t_insole)

# Optional coloring for visualization
insole_mesh_aligned.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow insole
arch_mesh.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta arch

# Visualize arch mesh + transformed original insole mesh together
# NOTE: mesh_show_back_face=True helps for thin sheet-like meshes.
o3d.visualization.draw_geometries(
    [arch_mesh, insole_mesh_aligned, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
    window_name="arch_mesh + insole_mesh_aligned",
    width=1400,
    height=900,
    mesh_show_back_face=True
)


# Visualize
# o3d.visualization.draw_geometries([pcd4_vis, pcd5_vis],
#                                   window_name="Red=Plane, Blue=Non-plane")
#
# o3d.visualization.draw_geometries([plane_pcd, non_plane_pcd, pcd4_vis, pcd5_vis],
#                                   window_name="Red=Plane, Blue=Non-plane")
#
# o3d.visualization.draw_geometries([pcd1, pcd5_vis],
#                                   window_name="foot vs arch")



clean_up()
close_all_plots()

"""
breakpoint()
































### ------------------------ <<<<<<<<<<<<<<<<<<<< ------------------------------------------


# get rotation right for pcd2
pca_result2 = compute_pca_orientation(filtered_pcd2)
print("\n--- Aligning Major Axis to Y-axis ---PCD2")
aligned_to_y_bot2 = align_major_axis_to_axis(filtered_pcd2, target_axis='y', pca_result=pca_result2)
aligned_to_y2 = align_major_axis_to_axis(cut_pcd2, target_axis='y', pca_result=pca_result2)

### further rotation
# Convert 5 degrees to radians

phone_rotation_degree = 0
#print("phone_rotation_degree: %.2f", phone_rotation_degree)
print("---------------phone_rotation_degree------------------")
print(f"phone_rotation_degree: {phone_rotation_degree:.2f}")
theta = np.deg2rad(phone_rotation_degree)
#breakpoint()

# Define rotation matrix around Z-axis
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

# Rotate the point cloud around its center
aligned_to_y2.rotate(R, center=aligned_to_y2.get_center())  # or use aligned_to_y2.get_center() if you want to rotate around its center




aligned_to_y_bot2.paint_uniform_color([0, 0.5, 0])
aligned_to_y2.paint_uniform_color([1, 0.5, 0])

o3d.visualization.draw_geometries(
            [aligned_to_y2, aligned_to_y_bot2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
            window_name="Aligned to Y-axis: phone",
            width=1400,
            height=900
        )



# Example usage:
if __name__ == "__main__":
    # Load two point clouds
    pcd1 = aligned_to_y1 #o3d.io.read_point_cloud("pointcloud1.ply")
    pcd2 = aligned_to_y2 #o3d.io.read_point_cloud("pointcloud2.ply")

    # Check original bounds
    get_bounding_box_info(pcd1, "Point Cloud 1 (Original)")
    get_bounding_box_info(pcd2, "Point Cloud 2 (Original)")

    # Method 1: Align both to the same minimum
    print("\n--- Method 1: Align both to same minimum ---")
    #aligned_pcd1, aligned_pcd2 = align_min_xyz(pcd1.clone(), pcd2.clone())
    aligned_pcd1, aligned_pcd2 = align_min_xyz(pcd1, pcd2)

    get_bounding_box_info(aligned_pcd1, "Point Cloud 1 (Aligned)")
    get_bounding_box_info(aligned_pcd2, "Point Cloud 2 (Aligned)")

    # Visualize
    aligned_pcd1.paint_uniform_color([1, 0, 0])  # Red
    aligned_pcd2.paint_uniform_color([0, 0, 1])  # Blue
    #o3d.visualization.draw_geometries([aligned_pcd1, aligned_pcd2])
    o3d.visualization.draw_geometries(
        [aligned_pcd1, aligned_pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
        window_name="Aligned stationary and phone",
        width=1400,
        height=900
    )

#breakpoint()

### -------------- get the metrics of the point cloud -----------------

# If using Open3D PointCloud objects, convert them to numpy arrays:
# import open3d as o3d
# If pc1 and pc2 are Open3D PointCloud objects:
pc1_points = np.asarray(aligned_to_y1.points)
pc2_points = np.asarray(aligned_to_y2.points)

# Compute AABBs
min1, max1, dims1, center1 = compute_aabb(pc1_points)
min2, max2, dims2, center2 = compute_aabb(pc2_points)

# Get corners
corners1 = get_aabb_corners(min1, max1)
corners2 = get_aabb_corners(min2, max2)

# Visualization
fig = plt.figure(figsize=(14, 6))

# 3D plot
ax1 = fig.add_subplot(121, projection='3d')

# Plot point clouds
ax1.scatter(*pc1_points.T, c='blue', alpha=0.3, s=1, label='Point Cloud 1')
ax1.scatter(*pc2_points.T, c='red', alpha=0.3, s=1, label='Point Cloud 2')

# Plot AABBs
plot_aabb(ax1, corners1, 'blue', 'AABB 1')
plot_aabb(ax1, corners2, 'red', 'AABB 2')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Point Clouds with Axis-Aligned Bounding Boxes')
ax1.legend()
ax1.set_aspect('equal')

# Dimensions comparison
ax2 = fig.add_subplot(122)

categories = ['X (Width)', 'Y (Length)', 'Z (Height)']
x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width / 2, dims1, width, label='Point Cloud 1', color='blue', alpha=0.2)
bars2 = ax2.bar(x + width / 2, dims2, width, label='Point Cloud 2', color='red', alpha=0.7)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Dimension')
ax2.set_ylabel('Size')
ax2.set_title('AABB Dimensions Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Print dimensions
print("Point Cloud 1 AABB Dimensions:")
print(f"  X (Width): {dims1[0]:.3f}")
print(f"  Y (Length):  {dims1[1]:.3f}")
print(f"  Z (Height): {dims1[2]:.3f}")
print(f"  Center: ({center1[0]:.3f}, {center1[1]:.3f}, {center1[2]:.3f})")
print(f"  Min corner: ({min1[0]:.3f}, {min1[1]:.3f}, {min1[2]:.3f})")
print(f"  Max corner: ({max1[0]:.3f}, {max1[1]:.3f}, {max1[2]:.3f})")

print("\nPoint Cloud 2 AABB Dimensions:")
print(f"  X (Width): {dims2[0]:.3f}")
print(f"  Y (Length):  {dims2[1]:.3f}")
print(f"  Z (Height): {dims2[2]:.3f}")
print(f"  Center: ({center2[0]:.3f}, {center2[1]:.3f}, {center2[2]:.3f})")
print(f"  Min corner: ({min2[0]:.3f}, {min2[1]:.3f}, {min2[2]:.3f})")
print(f"  Max corner: ({max2[0]:.3f}, {max2[1]:.3f}, {max2[2]:.3f})")

### breakpoint()
### get the girth data
# Create two slightly different cylinders
pc1 = np.asarray(aligned_to_y1.points) #create_cylinder_cloud(radius=1.0, height=5.0, num_points=5000, noise=0.02)
pc2 = np.asarray(aligned_to_y2.points) #create_cylinder_cloud(radius=1.02, height=5.0, num_points=5000, noise=0.02)

# Initialize slicer
slicer = PointCloudSlicer(tolerance=1)   ### all the points within 1 mm near the plane

# Define cutting plane (horizontal plane at z=2.5)
plane_origin = np.array([0, 0, 0])     ### points on the plane
plane_normal = np.array([0, 1, 0])     ### normal vector of plane

# Compare girths using CONVEX HULL method
results = slicer.compare_girths(pc1, pc2, plane_origin, plane_normal,
                                method='convex_hull')

print("Girth Comparison Results (Convex Hull Method):")
print(f"  Point Cloud 1 Girth: {results['girth1']:.4f}")
print(f"  Point Cloud 2 Girth: {results['girth2']:.4f}")
print(f"  Difference: {results['difference']:.4f}")
print(f"  Percent Change: {results['percent_change']:.2f}%")

# Dense boundary method - better matches actual data points
results_dense = slicer.compare_girths(pc1, pc2, plane_origin, plane_normal,
                                      method='dense_boundary')

print("\nGirth Comparison Results (Dense Boundary Method - Better Match):")
print(f"  Point Cloud 1 Girth: {results_dense['girth1']:.4f}")
print(f"  Point Cloud 2 Girth: {results_dense['girth2']:.4f}")
print(f"  Difference: {results_dense['difference']:.4f}")
print(f"  Percent Change: {results_dense['percent_change']:.2f}%")

# Also calculate with fitted_curve method for comparison
results_fitted = slicer.compare_girths(pc1, pc2, plane_origin, plane_normal,
                                       method='fitted_curve')

print("\nGirth Comparison Results (Fitted Curve Method - Smoothest):")
print(f"  Point Cloud 1 Girth: {results_fitted['girth1']:.4f}")
print(f"  Point Cloud 2 Girth: {results_fitted['girth2']:.4f}")
print(f"  Difference: {results_fitted['difference']:.4f}")
print(f"  Percent Change: {results_fitted['percent_change']:.2f}%")

# Visualize 3D with cutting plane - shows WHOLE point cloud
print("\n3D Visualization with cutting plane (showing whole point cloud)...")
_, _, slice1_idx = slicer.slice_with_plane(pc1, plane_origin, plane_normal)
_, _, slice2_idx = slicer.slice_with_plane(pc2, plane_origin, plane_normal)

# Side-by-side 3D comparison - WHOLE point clouds visible
fig_3d = slicer.visualize_comparison_3d(pc1, pc2, plane_origin, plane_normal,
                                        slice1_idx, slice2_idx)
plt.savefig('3d_comparison_with_plane.png', dpi=150, bbox_inches='tight')
print("3D visualization saved as '3d_comparison_with_plane.png'")

# 2D slice profiles with multiple methods
fig2, axes = plt.subplots(2, 3, figsize=(20, 13))

# Top row: Different methods for PC1
slicer.visualize_slice(results['slice1_2d'], results['boundary1'],
                       f"PC1 - Convex Hull (G={results['girth1']:.3f})",
                       ax=axes[0, 0])

slicer.visualize_slice(results_dense['slice1_2d'], results_dense['boundary1'],
                       f"PC1 - Dense Boundary (G={results_dense['girth1']:.3f})",
                       ax=axes[0, 1])

slicer.visualize_slice(results_fitted['slice1_2d'], results_fitted['boundary1'],
                       f"PC1 - Fitted Curve (G={results_fitted['girth1']:.3f})",
                       ax=axes[0, 2])

# Bottom row: Different methods for PC2
slicer.visualize_slice(results['slice2_2d'], results['boundary2'],
                       f"PC2 - Convex Hull (G={results['girth2']:.3f})",
                       ax=axes[1, 0])

slicer.visualize_slice(results_dense['slice2_2d'], results_dense['boundary2'],
                       f"PC2 - Dense Boundary (G={results_dense['girth2']:.3f})",
                       ax=axes[1, 1])

slicer.visualize_slice(results_fitted['slice2_2d'], results_fitted['boundary2'],
                       f"PC2 - Fitted Curve (G={results_fitted['girth2']:.3f})",
                       ax=axes[1, 2])

plt.tight_layout()
plt.savefig('2d_slice_profiles_comparison.png', dpi=150, bbox_inches='tight')
print("2D profiles comparison saved as '2d_slice_profiles_comparison.png'")
print("\nNote: Dense Boundary (middle) follows actual points better than Convex Hull (left)")
print("      Fitted Curve (right) is smoothest")

# Create overlay comparison plot (Plot 4)
fig4, axes4 = plt.subplots(1, 3, figsize=(24, 7))

# Overlay with Convex Hull
slicer.visualize_slice_overlay(results['slice1_2d'], results['boundary1'],
                               results['slice2_2d'], results['boundary2'],
                               results['girth1'], results['girth2'],
                               title="Overlay - Convex Hull",
                               ax=axes4[0])

# Overlay with Dense Boundary
slicer.visualize_slice_overlay(results_dense['slice1_2d'], results_dense['boundary1'],
                               results_dense['slice2_2d'], results_dense['boundary2'],
                               results_dense['girth1'], results_dense['girth2'],
                               title="Overlay - Dense Boundary",
                               ax=axes4[1])

# Overlay with Fitted Curve
slicer.visualize_slice_overlay(results_fitted['slice1_2d'], results_fitted['boundary1'],
                               results_fitted['slice2_2d'], results_fitted['boundary2'],
                               results_fitted['girth1'], results_fitted['girth2'],
                               title="Overlay - Fitted Curve",
                               ax=axes4[2])

plt.tight_layout()
plt.savefig('overlay_comparison.png', dpi=150, bbox_inches='tight')
print("Overlay comparison saved as 'overlay_comparison.png'")

plt.show()

### breakpoint()


### ------- new method of comparison

# Example usage:
if __name__ == "__main__":
    # # # Example: Create two similar meshes for demonstration
    # mesh1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # mesh1.compute_vertex_normals()
    #
    # # Create slightly different mesh
    # mesh2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # mesh2.compute_vertex_normals()
    # vertices = np.asarray(mesh2.vertices)
    # vertices += np.random.normal(0, 0.01, vertices.shape)  # Add small noise
    # mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    #
    # # # Method 1: GUI with side panel (more complex but has legend in window)
    # # print("Method 1: Custom GUI with legend panel")
    # # compare_point_clouds_with_legend(mesh1, mesh2, cmap_name='jet')
    #
    # # Method 2: Simple viewer with detailed console output (recommended)
    # print("\nMethod 2: Simple viewer with console stats")
    # simple_comparison_with_stats(mesh1, mesh2, cmap_name='jet')

    # Or use your own meshes:
    mesh1 = aligned_pcd1 #o3d.io.read_triangle_mesh("your_mesh1.ply")
    mesh2 = aligned_pcd2 #o3d.io.read_triangle_mesh("your_mesh2.ply")
    compare_point_clouds_with_legend(mesh1, mesh2, cmap_name='jet')
    simple_comparison_with_stats(mesh1, mesh2, cmap_name='jet')

#breakpoint()

### visualize the difference
# Example usage:
if __name__ == "__main__":
    # Load your point clouds
    pcd1 = aligned_pcd1 #o3d.io.read_point_cloud("pointcloud1.ply")
    pcd2 = aligned_pcd2 #o3d.io.read_point_cloud("pointcloud2.ply")

    # Method 1: Simple transparency simulation
    print("Method 1: Simple transparency")
    visualize_with_transparency(pcd1, pcd2,
                                color1=[1, 0, 0], color2=[0, 0, 1],
                                alpha1=0.2, alpha2=0.2)

    # # Method 2: Advanced renderer (true transparency)
    # print("\nMethod 2: Advanced renderer with transparency")
    # visualize_with_custom_renderer(pcd1, pcd2)

    # Method 3: Difference heatmap
    print("\nMethod 3: Difference heatmap")
    visualize_difference_heatmap(pcd1, pcd2)

    # Method 4: Alternating/mixed points
    print("\nMethod 4: Mixed points")
    visualize_alternating(pcd1, pcd2, sample_rate=0.5)





#breakpoint()


# # show the results,  Previous code

# setup up cut distance for better visualization
z_threshold = 100 # unit mm
mobile_mesh_cut = scan_o3d_mesh # mesh_meas.cut_mesh_at_z(scan_o3d_mesh, z_threshold)
stationary_mesh_cut = stationary_o3d_mesh # mesh_meas.cut_mesh_at_z(stationary_o3d_mesh, z_threshold)

# o3d.visualization.draw_geometries([mobile_mesh_cut,stationary_mesh_cut])

# ############ comparing meshes ############
# overlay_meshes(scan_o3d_mesh,stationary_o3d_mesh,alpha=0.1,copy_inputs=False)
# compare_meshes(stationary_o3d_mesh,scan_o3d_mesh)

compare_meshes_transparent(stationary_o3d_mesh,scan_o3d_mesh, alpha=0.05)
#compare_meshes_transparent(stationary_mesh_cut,mobile_mesh_cut, alpha=0.05)
#compare_meshes_transparent(aligned_pcd1,aligned_pcd2, alpha=0.15)
# mesh_common_methods.compare_ply_files(stationary_o3d_mesh, scan_o3d_mesh)
# mesh_common_methods.compare_ply_files(scan_mesh_path,stationary_scan_path,side_by_side=False)

# o3d.visualization.draw_geometries([scan_o3d_mesh, stationary_o3d_mesh])
print('finished comparison')
### ----------- end of main code --------------

"""