import numpy as np
from typing import Any
import open3d as o3d
import open3d.core as o3c


class TensorMap(dict):

    def __init__(self):
        super().__init__()

    def __setattr__(self, key: str, value: Any) -> None:
        sup = super()
        sup.__setitem__(key, value)

    def __getattr__(self, key: str) -> None:
        sup = super()
        if sup.__contains__(key):
            return sup.__getitem__(key)
        else:
            return None


if __name__ == "__main__":
    tm = TensorMap()
    tm.normals = 100
    print(tm.normals)
    print(tm.points)

    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3c.Tensor.ones((0, 3), o3c.float32)
    pcd.point.colors = o3c.Tensor.ones((2, 3), o3c.float32)
    pcd.point.colors = np.ones((4, 3), dtype=np.float32)
    print(pcd.point.colors)
    print(pcd.point)
