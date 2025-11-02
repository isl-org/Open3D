import numpy as np
import os
import open3d as o3d

def project_points_onto_mesh(points: np.ndarray,
                             direction: np.ndarray,
                             target_mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    使用 ray casting 将多个点沿着指定方向投射到 target_mesh 上。
    :param points: (N, 3) numpy array, 每个点是射线起点
    :param direction: (3,) numpy array, 所有射线的统一方向
    :param target_mesh: 被射线投射的目标 TriangleMesh
    :return: 命中的新点 (N, 3)，若未命中则保留原点
    """
    # 转为 tensor mesh
    target_t = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(target_t)

    # 构造射线数组 [N, 6] -> 每行 [x, y, z, dx, dy, dz]
    rays_np = np.hstack((points, np.tile(direction, (points.shape[0], 1)))).astype(np.float32)
    rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)

    # 执行 ray cast
    ans = scene.cast_rays(rays)
    t_hits = ans['t_hit'].numpy()

    # 命中的点：origin + t * direction
    hit_mask = np.isfinite(t_hits)
    projected_points = np.copy(points)
    projected_points[hit_mask] += direction * t_hits[hit_mask][:, np.newaxis]

    return projected_points