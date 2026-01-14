import open3d as o3d
import numpy as np

# Create a simple point cloud for example
points = np.random.rand(100, 3)  # 100 random points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Check that the method exists
print("Methods of PointCloud:", dir(pcd))
print("Does smooth exist?", hasattr(pcd, "smooth"))

# Store BEFORE smoothing
before_points = np.asarray(pcd.points)
print("BEFORE smoothing (first 5 points):\n", before_points[:5])

# Run smoothing
pcd.smooth(method="MLS", radius=0.05, k=10)

# Store AFTER smoothing
after_points = np.asarray(pcd.points)
print("AFTER smoothing (first 5 points):\n", after_points[:5])

# Compute difference
diff = after_points - before_points
print("Difference (first 5 points):\n", diff[:5])
print("Magnitude of movement (first 5 points):\n", np.linalg.norm(diff[:5], axis=1))

# Show point cloud
o3d.visualization.draw_geometries([pcd])
