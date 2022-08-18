import numpy as np
import open3d as o3d
import open3d.core as o3c

if __name__ == "__main__":
    pcd = o3d.t.geometry.PointCloud()
    print(pcd.point)
    # pcd.point.colors = o3c.Tensor.ones((2, 3), o3c.float32)
    # pcd.point.colors = np.ones((4, 3), dtype=np.float32)
    # print(pcd.point.colors)
    # print(pcd.point)
