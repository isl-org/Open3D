import open3d as o3d
import numpy as np

def get_line(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.5, max_nn=30))
    normals = np.asarray(pcd.normals)

    # Build KD-tree for neighborhood searches
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    cliff_scores = []
    for i in range(len(points)):
        # Find neighbors
        [k, idx, _] = kdtree.search_radius_vector_3d(points[i], 1.0)

        if k < 5:
            cliff_scores.append(0)
            continue

        neighbor_points = points[idx[1:], :]
        neighbor_normals = normals[idx[1:], :]

        # Height variation (Z-direction)
        height_std = np.std(neighbor_points[:, 2])

        # Normal variation (direction change)
        normal_variation = 1 - np.abs(np.dot(normals[i], neighbor_normals.T).mean())

        # Verticality of normal (cliffs are steep)
        verticality = np.abs(normals[i][2])  # How horizontal the normal is

        # Combined score
        cliff_score = height_std * normal_variation * (1 - verticality)
        cliff_scores.append(cliff_score)

    # Extract cliff points
    cliff_scores = np.array(cliff_scores)
    threshold = np.percentile(cliff_scores, 98)
    cliff_indices = np.where(cliff_scores > threshold)[0]

    # Visualize
    cliff_pcd = pcd.select_by_index(cliff_indices)
    cliff_pcd.paint_uniform_color([1, 0, 0])  # Red for cliff edge
    o3d.visualization.draw_geometries([cliff_pcd.paint_uniform_color([1.0, 1.0, 0.0])], mesh_show_back_face=True)

    return cliff_pcd




