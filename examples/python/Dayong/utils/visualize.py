import open3d as o3d
import sys
import os
from pathlib import Path


# Load and visualize an STL file in point cloud format
def visualize_stl(stl_path: str):
    stl_path = Path(stl_path)

    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh from: {stl_path}")

    # Ensure normals for proper shading
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    n_points = 10000 * 18  ### number of points to analyze
    pcd = mesh.sample_points_uniformly(number_of_points=n_points) # Convert the mesh to point cloud

    o3d.visualization.draw_geometries([
        pcd.paint_uniform_color([1, 0, 0]),
        # pcd2_points.paint_uniform_color([0, 1, 0]),
    ], window_name="Viewing STL file in Point Cloud Format")

if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    parent_dir = cur_dir.parent

    file_name = 'hd-0215.stl'
    file_dir = os.path.join(parent_dir, "scans", "STLs", "iPhoneScans", file_name)
    visualize_stl(file_dir)