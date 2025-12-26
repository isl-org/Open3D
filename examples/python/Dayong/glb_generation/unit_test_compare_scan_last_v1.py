### mian code starts from line 2052

import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.colors import Normalize
from open3d.visualization import gui as gui
from open3d.visualization import rendering as rendering


# from scripts.mesh_process import mesh_measure_methods
# from scripts.mesh_process import mesh_smooth_methods
# from scripts.mesh_process import mesh_common_methods
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering


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
        Blue : Small differences
        Green: Medium differences
        Yellow: Large differences
        Red  : Largest differences"""
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
    dists_mm = dists  # * 1  # Change to dists * 1 if already in mm

    # Apply colormap
    print("Applying colormap...")
    min_dist = dists_mm.min()
    max_dist = dists_mm.max()
    norm = Normalize(vmin=min_dist, vmax=max_dist)

    try:
        cmap = plt.colormaps.get_cmap(cmap_name)
    except:
        cmap = plt.cm.get_cmap(cmap_name)

    ### set color normalization as
    # dist_mm[i] > 12, set as 12
    # goal
    # example 1:  all distance within 4 mm, everything shown as blue,
    # example 2:  all distance > 12 mm,  everything shown as red
    ### distance values: [0 4]mm blue, [4 8]mm middle, [12 and above], red
    ### new way to normalized dists_mm

    ###

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

    return pcd1_points


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
    B.paint_uniform_color([0.3, 0.5, 1.0])  # reddish [1.0, 0.3, 0.2]

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
            {"name": "Mesh B", "geometry": B, "material": matB},
            {"name": "Axes", "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)},
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
    x_min1 = np.asarray(m1.vertices)[:, 0].mean(axis=0)  # .min()
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


def align_major_axis_to_axis(pcd, target_axis='y', pca_result=None):
    """
    Align the major principal axis (PC1) to a specific world axis

    Args:
        pcd: Original point cloud
        target_axis: Target axis ('x', 'y', or 'z')
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

    return pcd_aligned


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
from scipy.spatial import ConvexHull, distance_matrix
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA


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


import open3d as o3d
import numpy as np


def resize_point_cloud_xy(pcd, target_width_x, target_length_y):
    """
    Resize a point cloud to specific X and Y dimensions while maintaining the center point.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Input point cloud to resize
    target_width_x : float
        Desired width along X axis
    target_length_y : float
        Desired length along Y axis

    Returns:
    --------
    open3d.geometry.PointCloud
        Resized point cloud
    """
    # Get the current points
    points = np.asarray(pcd.points).copy()

    # Calculate current bounding box
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Calculate current center
    center = (min_bound + max_bound) / 2.0

    # Calculate current dimensions
    current_width_x = max_bound[0] - min_bound[0]
    current_length_y = max_bound[1] - min_bound[1]

    # Calculate scaling factors
    scale_x = target_width_x / current_width_x if current_width_x > 0 else 1.0
    scale_y = target_length_y / current_length_y if current_length_y > 0 else 1.0

    # Translate to origin (center at 0,0,0)
    points_centered = points - center

    # Apply scaling only to X and Y axes
    points_centered[:, 0] *= scale_x  # Scale X
    points_centered[:, 1] *= scale_y  # Scale Y
    # Z remains unchanged

    # Translate back to original center
    points_final = points_centered + center

    # Create new point cloud
    pcd_resized = o3d.geometry.PointCloud()
    pcd_resized.points = o3d.utility.Vector3dVector(points_final)

    # Copy colors if they exist
    if pcd.has_colors():
        pcd_resized.colors = pcd.colors

    # Copy normals if they exist
    if pcd.has_normals():
        pcd_resized.normals = pcd.normals

    return pcd_resized


### ------------- end of function region --------------------------


### ---------------------------------------------------------------


### ------------------------------------------------
### ----------- main code ---------------------
# *********** load mesh from file ***********
# from scripts.post_process.clean_up import clean_up

foot_id = "left"

### Phase 1: find the optimal match shoe last

### Phase 2: visualize the fitting between optimal last and foot (or ideal last)

'''
Step3b: align foot and last, then cut with the sane surface then compare the girth 
Step 4: based on foot length/width, scale the adjacent last to get the ideal last. 
Step 5a: compute visualization of foot and last
Step 5b: compare ideal last and optimal last, and show visualization 
Step 5c: decide whether to show foot glb
Step 6: output the glb file with color information 


*/'''

import os
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
dayong_dir = cur_dir.parent
stationary_case_path = os.path.join(dayong_dir, "scans", "STLs")
mobile_case_path = os.path.join(dayong_dir, "scans", "STLs")

# stationary_case_path = os.path.join("../", base_dir, "scans","GLBs")
# mobile_case_path = os.path.join("../", base_dir, "scans", "GLBs")

stationary_case = '/41+4'  # 'Stride_41'
# stationary_case_path = '/Users/yujunwang/Documents/2025Folder/Coding Work/customization1.0/Custom V1 last library/'
# mobile_case_path = '/Users/yujunwang/Documents/2025Folder/Coding Work/scanData/Yujun_bare_bottomR/'


# Bryan/CAB9/DepthData
# TC/2AC2/DepthData
# Justin/24Apr03
# Jian/4495

# load methods
# mesh_meas = mesh_measure_methods.MeshMeasureMethods()

# load stationary mesh
if foot_id == 'left':
    stl_foot_id = 'L'
else:
    stl_foot_id = 'R'
# stationary_scan_path = file_path_base+"right_foot_mesh.stl"
# stationary_scan_path= '/Users/nic_gong/Documents/StationaryScanData/StationaryScanValidationData/Jian_Gong_Fullweight4_550013_000027/Jian_Gong Fullweight4_550013_000027_L.stl'

stationary_scan_path = stationary_case_path + stationary_case + '.stl'
print(stationary_scan_path)
stationary_o3d_mesh = o3d.io.read_triangle_mesh(stationary_scan_path)
# scale stl dimension from mm to m
# stationary_o3d_mesh.scale(1e-3, center=stationary_o3d_mesh.get_center())

### load the file
# load mobile scanned mesh
# scan_mesh_path = mobile_case_path + '_'+stl_foot_id+".stl"
# scan_mesh_path = mobile_case_path + foot_id +"_foot_mesh_refine.stl"
scan_mesh_path = mobile_case_path + "/" + foot_id + "_foot_mesh.stl"

scan_o3d_mesh = o3d.io.read_triangle_mesh(scan_mesh_path)  ### foot 3D model from phone scan
scan_o3d_mesh.scale(1000.0, center=scan_o3d_mesh.get_center())

# Align point clouds first if needed
# pcd2.transform(transformation_matrix)
pcd1 = stationary_o3d_mesh  ### 3D last data
pcd2 = scan_o3d_mesh  ### 3D foot data

n_points = 10000 * 10  # 10000 * 18     ### number of points to analyze
# Convert meshes to point clouds
if isinstance(pcd1, o3d.geometry.TriangleMesh):
    pcd1_points = pcd1.sample_points_uniformly(number_of_points=n_points)
else:
    pcd1_points = pcd1

if isinstance(pcd2, o3d.geometry.TriangleMesh):
    pcd2_points = pcd2.sample_points_uniformly(number_of_points=n_points)
else:
    pcd2_points = pcd2

# visualize the results
# show the results
o3d.visualization.draw_geometries([
    pcd1_points.paint_uniform_color([0, 0, 1]),  # blue
    pcd2_points.paint_uniform_color([1, 0, 0]),  # red
], window_name="check read STL files Shoe Last and Phone Scan Foot")

### ------ cut the foot to get the oritentation right
# Method 1: Filter from min Z with specific distance
print("\n--- Method 1: Filter from min Z + distance ---")
distance = 20.0  # [mm] Keep 2.0 units from the bottom
filtered_pcd1 = filter_from_min_z_with_distance(pcd1_points, distance_from_min=distance)
filtered_pcd2 = filter_from_min_z_with_distance(pcd2_points, distance_from_min=distance)

distance = 100.0  # [mm] Keep 2.0 units from the bottom
cut_pcd1 = filter_from_min_z_with_distance(pcd1_points, distance_from_min=distance)
cut_pcd2 = filter_from_min_z_with_distance(pcd2_points, distance_from_min=distance)

# # Visualize
# print("------pcd1------")
# visualize_filtered_comparison(pcd1_points, filtered_pcd1)
# print("------pcd2------")
# visualize_filtered_comparison(pcd2_points, filtered_pcd2)


### ------------ rotate the orientation ----------------------
# get rotation right for pcd1, shoe last
pca_result1 = compute_pca_orientation(filtered_pcd1)
print("\n--- Aligning Major Axis to Y-axis ---PCD1")
aligned_to_y_bot1 = align_major_axis_to_axis(filtered_pcd1, target_axis='y', pca_result=pca_result1)
aligned_to_y1 = align_major_axis_to_axis(cut_pcd1, target_axis='y', pca_result=pca_result1)

aligned_to_y_bot1.paint_uniform_color([0, 0.5, 0])
aligned_to_y1.paint_uniform_color([0, 0, 1])

o3d.visualization.draw_geometries(
    [aligned_to_y1, aligned_to_y_bot1, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
    window_name="Aligned to Y-axis: last",
    width=1400,
    height=900
)

# get rotation right for pcd2, foot 3D model
pca_result2 = compute_pca_orientation(filtered_pcd2)
print("\n--- Aligning Major Axis to Y-axis ---PCD2")
aligned_to_y_bot2 = align_major_axis_to_axis(filtered_pcd2, target_axis='y', pca_result=pca_result2)
aligned_to_y2 = align_major_axis_to_axis(cut_pcd2, target_axis='y', pca_result=pca_result2)

### further rotation
# Convert 5 degrees to radians

phone_rotation_degree = 0
# print("phone_rotation_degree: %.2f", phone_rotation_degree)
print("---------------phone_rotation_degree------------------")
print(f"phone_rotation_degree: {phone_rotation_degree:.2f}")
theta = np.deg2rad(phone_rotation_degree)
# breakpoint()

# Define rotation matrix around Z-axis
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

# Rotate the point cloud around its center
aligned_to_y2.rotate(R,
                     center=aligned_to_y2.get_center())  # or use aligned_to_y2.get_center() if you want to rotate around its center

aligned_to_y_bot2.paint_uniform_color([0, 0.5, 0])
aligned_to_y2.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries(
    [aligned_to_y2, aligned_to_y_bot2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
    window_name="Aligned to Y-axis: foot",
    width=1400,
    height=900
)

# Load two point clouds
pcd1 = aligned_to_y1  # o3d.io.read_point_cloud("pointcloud1.ply")
pcd2 = aligned_to_y2  # o3d.io.read_point_cloud("pointcloud2.ply")

# Check original bounds
get_bounding_box_info(pcd1, "Point Cloud 1 (Original)")
get_bounding_box_info(pcd2, "Point Cloud 2 (Original)")

# Method 1: Align both to the same minimum
print("\n--- Method 1: Align both to same minimum ---")
# aligned_pcd1, aligned_pcd2 = align_min_xyz(pcd1.clone(), pcd2.clone())
aligned_pcd1, aligned_pcd2 = align_min_xyz(pcd1, pcd2)

get_bounding_box_info(aligned_pcd1, "Point Cloud 1 (Aligned)")
get_bounding_box_info(aligned_pcd2, "Point Cloud 2 (Aligned)")

# Visualize final aligned comparison
aligned_pcd1.paint_uniform_color([0, 0, 1])  # last, blue Red
aligned_pcd2.paint_uniform_color([1, 0, 0])  # foot, red  Blue
# o3d.visualization.draw_geometries([aligned_pcd1, aligned_pcd2])
o3d.visualization.draw_geometries(
    [aligned_pcd1, aligned_pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
    window_name="Aligned last and foot",
    width=1400,
    height=900
)

# breakpoint()

### -------------- get the metrics of the point cloud -----------------

# If using Open3D PointCloud objects, convert them to numpy arrays:
# import open3d as o3d
# If pc1 and pc2 are Open3D PointCloud objects:

# pc1_points = np.asarray(aligned_to_y1.points)
# pc2_points = np.asarray(aligned_to_y2.points)

pc1_points = np.asarray(aligned_pcd1.points)
pc2_points = np.asarray(aligned_pcd2.points)

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
ax1.scatter(*pc1_points.T, c='blue', alpha=0.3, s=1, label='Last Point Cloud')
ax1.scatter(*pc2_points.T, c='red', alpha=0.3, s=1, label='Foot Point Cloud')

# Plot AABBs
plot_aabb(ax1, corners1, 'blue', 'Last AABB 1')
plot_aabb(ax1, corners2, 'red', 'Foot AABB 2')

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

bars1 = ax2.bar(x - width / 2, dims1, width, label='Last Point Cloud', color='blue', alpha=1)
bars2 = ax2.bar(x + width / 2, dims2, width, label='Foot Point Cloud', color='red', alpha=1)

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

ideal_last_length = dims2[1] + 15
ideal_last_width = dims2[0]

# o3d.visualization.draw_geometries(
#     [aligned_pcd1, aligned_pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
#     window_name="Aligned last and foot",
#     width=1400,
#     height=900
# )

ideal_last = resize_point_cloud_xy(aligned_pcd1, target_width_x=ideal_last_width, target_length_y=ideal_last_length)

aligned_pcd1, ideal_last = align_min_xyz(aligned_pcd1, ideal_last)  ### aligned based min values in [x,y,z]

# # Verify the results
# points_resized = np.asarray(ideal_last.points)
# print(
#     f"New X range: {points_resized[:, 0].min():.3f} to {points_resized[:, 0].max():.3f}, width: {points_resized[:, 0].max() - points_resized[:, 0].min():.3f}")
# print(
#     f"New Y range: {points_resized[:, 1].min():.3f} to {points_resized[:, 1].max():.3f}, length: {points_resized[:, 1].max() - points_resized[:, 1].min():.3f}")
# print(f"Center: {(points_resized.min(axis=0) + points_resized.max(axis=0)) / 2.0}")
ideal_last.paint_uniform_color([1, 1, 0])  # foot, red  Blue

o3d.visualization.draw_geometries(
    [aligned_pcd1, ideal_last],
    window_name="mesh 1 and 2",
    width=1400,
    height=900
)

# breakpoint()


### get the girth data
# Create two slightly different cylinders
# pc1 = np.asarray(aligned_to_y1.points) #create_cylinder_cloud(radius=1.0, height=5.0, num_points=5000, noise=0.02)
# pc2 = np.asarray(aligned_to_y2.points) #create_cylinder_cloud(radius=1.02, height=5.0, num_points=5000, noise=0.02)

pc1 = np.asarray(aligned_pcd1.points)  # np.asarray(aligned_to_y1.points)
pc2 = np.asarray(aligned_pcd2.points)  # np.asarray(aligned_to_y2.points)

# Initialize slicer
slicer = PointCloudSlicer(tolerance=1)  ### all the points within 1 mm near the plane

# Define cutting plane (horizontal plane at z=2.5)
plane_origin = np.array([0, 0, 0])  ### points on the plane
plane_normal = np.array([0, 1, 0])  ### normal vector of plane

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

# breakpoint()


### ------- new method of comparison
# Example usage:
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
mesh1 = aligned_pcd1  # o3d.io.read_triangle_mesh("your_mesh1.ply")
# mesh2 = aligned_pcd2 #o3d.io.read_triangle_mesh("your_mesh2.ply")
mesh2 = ideal_last  # o3d.io.read_triangle_mesh("your_mesh2.ply")

o3d.visualization.draw_geometries(
    [mesh1, mesh2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
    window_name="mesh 1 and 2",
    width=1400,
    height=900
)

### comparison 1: transparent with differences using PointCloudViewerWithLegend

### note for Dayong
### this is the main comparison method for visualization
### return point cloud with color: color_pc_1
### Q1:  save "color_pc_1" as a GLB file with color information, find a software to check the color
### Q2:  color setting:  in Jet legend,  [0 4]mm  show as blue, [4 8]mm middle color; [8 beyond] show as red

color_pc_1 = compare_point_clouds_with_legend(mesh1, mesh2, cmap_name='jet')

o3d.visualization.draw_geometries(
    [color_pc_1, o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)],
    window_name="colored last 3D",
    width=1400,
    height=900
)

# Dayong starts
# Extract points and colors from Open3D point cloud

# =============================================================
# GLB export pipeline (Poisson-only): colored PointCloud -> mesh -> GLB
# 1) Poisson surface reconstruction (smooth + mostly watertight)
# 2) Optional hole filling (protect the main opening, e.g., shoe collar)
# 3) Transfer heatmap colors to mesh vertices (1-NN)
# 4) Export GLB via trimesh so Blender/MeshLab reliably show vertex colors
# =============================================================

import trimesh

# Optional: hole-filling backend (preferred)
try:
    import pymeshlab as ml

    _HAVE_PYMESHLAB = True
except Exception:
    ml = None
    _HAVE_PYMESHLAB = False


def _fill_holes_in_mesh(
        mesh: o3d.geometry.TriangleMesh,
        method: str = "auto",
        max_hole_size: int = 2000,
        verbose: bool = True,
        keep_largest_openings: int = 1,
        auto_avoid_large_openings: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Fill mesh holes (post-process) while keeping the main opening unfilled.

    Key idea:
      - Find boundary loops (edges used by exactly one triangle).
      - Keep the largest N loops open (e.g., shoe collar).
      - Optionally cap max_hole_size so backends cannot close that opening.
    """
    # Work on a copy (do not mutate caller mesh)
    m = o3d.geometry.TriangleMesh(mesh)
    method_l = (method or "auto").lower().strip()

    def _boundary_loops_edge_counts(_mesh: o3d.geometry.TriangleMesh):
        """Approx loop size by number of boundary edges (proxy for hole size)."""
        F = np.asarray(_mesh.triangles)
        if F.size == 0:
            return []

        # Build undirected edges from triangles and count occurrences
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        from collections import Counter
        cnt = Counter([tuple(e) for e in edges])

        # Boundary edges appear exactly once
        b_edges = [e for e, c in cnt.items() if c == 1]
        if not b_edges:
            return []

        # Build adjacency and find connected components
        adj = {}
        for a, b in b_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        visited = set()
        loop_sizes = []
        for start in adj.keys():
            if start in visited:
                continue
            stack = [start]
            comp_edges = set()
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                for nb in adj.get(v, []):
                    comp_edges.add(tuple(sorted((v, nb))))
                    if nb not in visited:
                        stack.append(nb)
            loop_sizes.append(len(comp_edges))
        return loop_sizes

    # Auto-cap max_hole_size to guarantee we never seal the largest opening(s)
    if auto_avoid_large_openings and keep_largest_openings and keep_largest_openings > 0:
        loop_sizes = sorted(_boundary_loops_edge_counts(m), reverse=True)
        if loop_sizes:
            k = min(int(keep_largest_openings), len(loop_sizes))
            smallest_kept = loop_sizes[k - 1]
            safe_cap = max(0, int(smallest_kept) - 1)
            if safe_cap < max_hole_size:
                if verbose:
                    print(
                        f"[HoleFill] boundary loops (edge counts): {loop_sizes[:5]}{'...' if len(loop_sizes) > 5 else ''}")
                    print(f"[HoleFill] keep_largest_openings={k} => cap max_hole_size {max_hole_size} -> {safe_cap}")
                max_hole_size = safe_cap

    if method_l in {"none", "off", "false"}:
        return m

    # Preferred backend: PyMeshLab
    if method_l in {"auto", "pymeshlab"} and _HAVE_PYMESHLAB:
        try:
            V = np.asarray(m.vertices, dtype=np.float64)
            F = np.asarray(m.triangles, dtype=np.int32)

            ms = ml.MeshSet()
            ms.add_mesh(ml.Mesh(V, F), "mesh")
            ms.apply_filter("meshing_close_holes", maxholesize=int(max_hole_size))

            V2 = ms.current_mesh().vertex_matrix()
            F2 = ms.current_mesh().face_matrix()

            out = o3d.geometry.TriangleMesh()
            out.vertices = o3d.utility.Vector3dVector(V2)
            out.triangles = o3d.utility.Vector3iVector(F2)
            out.remove_duplicated_vertices()
            out.remove_degenerate_triangles()
            out.remove_duplicated_triangles()
            out.remove_unreferenced_vertices()
            out.remove_non_manifold_edges()
            out.compute_vertex_normals()

            if verbose:
                print(
                    f"[HoleFill] PyMeshLab close_holes (max_hole_size={max_hole_size}, keep_largest_openings={keep_largest_openings})")
            return out
        except Exception as e:
            if verbose:
                print(f"[HoleFill] PyMeshLab failed; fallback. Reason: {e}")

    # Fallback: Open3D fill_holes (only if this Open3D build supports it)
    if method_l in {"auto", "open3d"} and hasattr(m, "fill_holes"):
        try:
            out = o3d.geometry.TriangleMesh(m).fill_holes(int(max_hole_size))
            out.remove_duplicated_vertices()
            out.remove_degenerate_triangles()
            out.remove_duplicated_triangles()
            out.remove_unreferenced_vertices()
            out.remove_non_manifold_edges()
            out.compute_vertex_normals()
            if verbose:
                print(
                    f"[HoleFill] Open3D fill_holes (max_hole_size={max_hole_size}, keep_largest_openings={keep_largest_openings})")
            return out
        except Exception as e:
            if verbose:
                print(f"[HoleFill] Open3D fill_holes failed. Reason: {e}")

    if verbose:
        print("[HoleFill] No hole filling performed (no backend available)")
    return m


def _nn_distance_stats(pcd: o3d.geometry.PointCloud):
    """NN spacing stats as a density proxy (used for auto params)."""
    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    if nn.size == 0:
        raise ValueError("Point cloud is empty; cannot compute NN stats.")
    return {"median": float(np.median(nn))}


def _auto_normal_radius_from_nn(pcd: o3d.geometry.PointCloud, multiplier: float = 4.0):
    """Normal radius ~ a few NN spacings (robust default)."""
    return max(1e-6, multiplier * _nn_distance_stats(pcd)["median"])


def _auto_poisson_depth_from_density(pcd: o3d.geometry.PointCloud, min_depth: int = 7, max_depth: int = 11):
    """Choose Poisson depth from density + object scale (higher = more detail)."""
    base = _nn_distance_stats(pcd)["median"]
    aabb = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(np.asarray(aabb.get_extent(), dtype=np.float64)))
    target = max(1e-6, 2.0 * base)
    depth = int(np.round(np.log2(max(diag / target, 1.0))) + 2)
    return int(np.clip(depth, min_depth, max_depth))


def colored_pcd_to_colored_mesh_glb(
        color_pc: o3d.geometry.PointCloud,
        glb_path="colored_mesh.glb",
        poisson_depth=None,  # None -> auto
        normal_radius=None,  # None -> auto
        density_trim_quantile=0.01,  # lower keeps more triangles
        smooth=True,
        smooth_iters=5,
        hole_fill=True,
        hole_fill_method="auto",  # "auto" | "pymeshlab" | "open3d" | "none"
        hole_max_size=2000,
        keep_largest_openings=1,
        auto_avoid_large_openings=True,
):
    """Poisson-only: colored PointCloud -> colored mesh -> GLB (vertex colors)."""

    # 0) Validate input
    if color_pc.is_empty():
        raise ValueError("color_pc is empty")
    if not color_pc.has_colors():
        raise ValueError("color_pc has no colors; please set color_pc.colors first")

    # 1) Copy input (do not mutate caller)
    pcd = o3d.geometry.PointCloud(color_pc)

    # 2) Normals (Poisson requires normals)
    if normal_radius is None:
        normal_radius = _auto_normal_radius_from_nn(pcd, multiplier=4.0)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=30)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass

    # 3) Poisson reconstruction
    if poisson_depth is None:
        poisson_depth = _auto_poisson_depth_from_density(pcd)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth)
    )

    # 3b) Trim low-density artifacts (floating sheets / spikes)
    densities = np.asarray(densities)
    if densities.size > 0:
        thr = np.quantile(densities, float(density_trim_quantile))
        mesh = mesh.select_by_index(np.where(densities >= thr)[0])

    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction failed; got empty mesh/triangles")

    # 4) Cleanup + light smoothing
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if smooth:
        try:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
            mesh.remove_unreferenced_vertices()
            mesh.remove_non_manifold_edges()
            mesh.compute_vertex_normals()
        except Exception:
            pass

    # 5) Hole filling (optional): Poisson is often watertight, but fill leftovers (keep collar open)
    if hole_fill:
        mesh = _fill_holes_in_mesh(
            mesh,
            method=str(hole_fill_method),
            max_hole_size=int(hole_max_size),
            verbose=True,
            keep_largest_openings=int(keep_largest_openings),
            auto_avoid_large_openings=bool(auto_avoid_large_openings),
        )

    # 6) Color transfer: point colors -> vertex colors (1-NN)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)

    vcols = np.zeros((len(mesh_vertices), 3), dtype=np.float64)
    for i, v in enumerate(mesh_vertices):
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        vcols[i] = pcd_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    mesh.compute_vertex_normals()

    # 7) Export GLB with vertex colors (Blender/MeshLab-friendly)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors)

    C255 = (np.clip(C, 0, 1) * 255).astype(np.uint8)
    if C255.shape[1] == 3:
        C255 = np.concatenate([C255, np.full((C255.shape[0], 1), 255, np.uint8)], axis=1)

    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=C255)
    tm.export(glb_path)
    print(f"[OK] Wrote GLB with vertex colors: {glb_path}")

    print("[AutoParams] normal_radius:", normal_radius)
    print("[AutoParams] poisson_depth:", poisson_depth, "density_trim_quantile:", density_trim_quantile)
    print("[AutoParams] hole_fill:", hole_fill, "hole_max_size:", hole_max_size,
          "keep_largest_openings:", keep_largest_openings, "auto_avoid_large_openings:", auto_avoid_large_openings)

    return mesh


# ---- Run: export GLB from heatmap point cloud ----
colored_mesh = colored_pcd_to_colored_mesh_glb(
    color_pc_1,
    glb_path="poisson_filled.glb",
    hole_fill=True,
    hole_fill_method="auto",
    hole_max_size=10000,
    keep_largest_openings=1,  # keep the shoe opening unfilled
    auto_avoid_large_openings=True,
)

# Dayong ends


# points = np.asarray(color_pc_1.points)
# colors = np.asarray(color_pc_1.colors)
#
# ### method 1
# # Convert colors from [0,1] to [0,255] if needed
# if colors.max() <= 1.0:
#     colors = (colors * 255).astype(np.uint8)
#
# # Create a trimesh point cloud
# cloud = trimesh.PointCloud(vertices=points, colors=colors)
#
# # Export as GLB
# cloud.export('old1.glb')
#
# ### method 2
# radii = [0.005, 0.01, 0.02, 0.04]
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     color_pc_1,
#     o3d.utility.DoubleVector(radii)
# )
#
# # Save as GLB (Open3D xxsupports glb/gltf export)
# o3d.io.write_triangle_mesh('old2.glb', mesh)


###
### comparison 2
simple_comparison_with_stats(mesh1, mesh2, cmap_name='jet')

# breakpoint()

### visualize the difference
# Example usage:
# Load your point clouds
pcd1 = aligned_pcd1  # o3d.io.read_point_cloud("pointcloud1.ply")
pcd2 = aligned_pcd2  # o3d.io.read_point_cloud("pointcloud2.ply")

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

# breakpoint()

# # show the results,  Previous code

# # setup up cut distance for better visualization
# z_threshold = 100 # unit mm
# mobile_mesh_cut = scan_o3d_mesh # mesh_meas.cut_mesh_at_z(scan_o3d_mesh, z_threshold)
# stationary_mesh_cut = stationary_o3d_mesh # mesh_meas.cut_mesh_at_z(stationary_o3d_mesh, z_threshold)
#
# # o3d.visualization.draw_geometries([mobile_mesh_cut,stationary_mesh_cut])
#
# # ############ comparing meshes ############
# # overlay_meshes(scan_o3d_mesh,stationary_o3d_mesh,alpha=0.1,copy_inputs=False)
# # compare_meshes(stationary_o3d_mesh,scan_o3d_mesh)
#
# # compare_meshes_transparent(stationary_o3d_mesh,scan_o3d_mesh, alpha=0.05)
# compare_meshes_transparent(stationary_mesh_cut,mobile_mesh_cut, alpha=0.05)
# #compare_meshes_transparent(aligned_pcd1,aligned_pcd2, alpha=0.15)
# # mesh_common_methods.compare_ply_files(stationary_o3d_mesh, scan_o3d_mesh)
# # mesh_common_methods.compare_ply_files(scan_mesh_path,stationary_scan_path,side_by_side=False)
#
# # o3d.visualization.draw_geometries([scan_o3d_mesh, stationary_o3d_mesh])

print('finished comparison')

# clean up and free memory
# clean_up()

### ----------- end of main code --------------
