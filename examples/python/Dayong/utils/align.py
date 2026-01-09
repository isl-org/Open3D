import numpy as np
import open3d as o3d
import copy


def align_xyz(target: o3d.geometry.PointCloud, source: o3d.geometry.PointCloud, in_place=False):
    """
    Align two point clouds so they have the same minimum x, y, z values

    Args:
        target: First point cloud
        source: Second point cloud
        in_place: If True, modify original point clouds; if False, create copies

    Returns:
        tuple: (aligned_pcd1, aligned_pcd2)
    """
    # Create copies if not modifying in place
    if not in_place:
        target = o3d.geometry.PointCloud(target)
        source = o3d.geometry.PointCloud(source)

    # Get points
    points1 = np.asarray(target.points)
    points2 = np.asarray(source.points)

    # Get minimum values for each
    min1 = points1.min(axis=0)
    min2 = points2.min(axis=0)

    print(f"PCD1 min: [{min1[0]:.4f}, {min1[1]:.4f}, {min1[2]:.4f}]")
    print(f"PCD2 min: [{min2[0]:.4f}, {min2[1]:.4f}, {min2[2]:.4f}]")

    # Calculate the overall minimum
    overall_min = np.minimum(min1, min2)

    print(f"Target min: [{overall_min[0]:.4f}, {overall_min[1]:.4f}, {overall_min[2]:.4f}]")

    # Translate both to have the same minimum
    translation1 = overall_min - min1
    translation2 = overall_min - min2

    target.translate(translation1)
    source.translate(translation2)

    print(f"Translation 1: [{translation1[0]:.4f}, {translation1[1]:.4f}, {translation1[2]:.4f}]")
    print(f"Translation 2: [{translation2[0]:.4f}, {translation2[1]:.4f}, {translation2[2]:.4f}]")

    return target, source