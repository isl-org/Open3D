import numpy as np
import open3d as o3d
import open3d.core as o3c


class WrongType():
    pass


if __name__ == "__main__":
    pcd = o3d.t.geometry.PointCloud()
    print(pcd.point)

    print("\n[Set new attribute with Open3D Tensor]")
    pcd.point.colors = o3c.Tensor.ones((2, 3), o3c.float32)
    print(pcd.point)

    print("\n[Set new attribute with Numpy array]")
    pcd.point.positions = np.ones((2, 3), np.float32)
    print(pcd.point)

    print("\n[Set existing attribute with wrong type]")
    try:
        pcd.point.positions = WrongType()
    except TypeError as e:
        print(f"Error: {e}")

    print("\n[Set new attribute with wrong type]")
    try:
        pcd.point.normals = WrongType()
    except TypeError as e:
        print(f"Error: {e}")

    print("\n[Get existing attribute]")
    print("colors:\n", pcd.point.colors)

    print("\n[Get unknown attribute]")
    try:
        print("normals:\n", pcd.point.normals)
    except KeyError as e:
        print(f"Error: {e}")

    print("\n[Get built-in functions or attributes]")
    print(f"primary_key: {pcd.point.primary_key}")

    print("\n[Set built-in functions or attributes]")
    try:
        pcd.point.primary_key = o3c.Tensor.ones((2, 3), o3c.float32)
    except KeyError as e:
        print(f"Error: {e}")
    print(pcd.point)

    print("\n[Using string as key should be avoided]")
    try:
        print(pcd.point["positions"])
    except RuntimeError as e:
        print(f"Error: {e}")

    del pcd.point.colors

    import ipdb
    ipdb.set_trace()
