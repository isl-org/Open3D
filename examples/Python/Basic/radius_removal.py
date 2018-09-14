import open3d as o3d
import numpy as np

pts = np.random.rand(100,3)
cloud = o3d.PointCloud()
cloud.points = o3d.Vector3dVector(pts)
new_cloud = o3d.radius_outlier_removal(cloud,10,0.2)
print(new_cloud)
