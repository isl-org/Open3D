import numpy as np
import open3d as o3d


def project_points_onto_mesh(points: np.ndarray,
                             direction: np.ndarray,
                             target_mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Use RayCasting to project points onto mesh
    Ignore points that missed
    """
    # Create Tensor Mesh
    target_t = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(target_t)

    # Construct ray casting array [N,6] -> [x,y,z,dx,dy,dz]
    rays_np = np.hstack((points, np.tile(direction, (points.shape[0], 1)))).astype(np.float32)
    rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)

    # Ray cast
    ans = scene.cast_rays(rays)
    t_hit = ans['t_hit'].numpy()

    # Hit the mask
    hit_mask = np.isfinite(t_hit)
    projected_points = np.copy(points)
    projected_points[hit_mask] += direction * t_hit[hit_mask][:, np.newaxis]

    print(f"[RayCast] Total: {len(points)}, Hit: {np.sum(hit_mask)}, Missed: {np.sum(~hit_mask)}")

    return projected_points, hit_mask