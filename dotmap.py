import numpy as np
import open3d as o3d
import open3d.core as o3c

if __name__ == "__main__":
    pcd = o3d.t.geometry.PointCloud()
    print(pcd.point)

    print("\n[Set new attribute with Open3D Tensor]")
    pcd.point.colors = o3c.Tensor.ones((2, 3), o3c.float32)
    print(pcd.point)

    print("\n[Set new attribute with Numpy array]")
    pcd.point.positions = np.ones((2, 3), np.float32)
    print(pcd.point)

    print("\n[Get existing attribute]")
    print("colors:\n", pcd.point.colors)

    print("\n[Get unknown attribute]")
    try:
        print("normals:\n", pcd.point.normals)
    except KeyError as e:
        print(f"Error: {e}")

    print("\n[Call other functions and attributes]")
    print(f"primary_key: {pcd.point.primary_key}")

    import ipdb
    ipdb.set_trace()
