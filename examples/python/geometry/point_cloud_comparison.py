import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsamples and computes FPFH features for global registration.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Performs RANSAC-based global registration for initial alignment.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def run_comparison(source_path, target_path, is_target_mesh=False):
    """
    Complete pipeline: Load -> Global Align -> Fine Align -> Heatmap.
    """
    # 1. Data Loading
    print(":: Loading datasets...")
    if is_target_mesh:
        mesh = o3d.io.read_triangle_mesh(target_path)
        mesh.compute_vertex_normals()
        # Sampling from mesh to allow distance computation
        target = mesh.sample_points_poisson_disk(number_of_points=50000)
    else:
        target = o3d.io.read_point_cloud(target_path)

    source = o3d.io.read_point_cloud(source_path)

    # 2. Pre-processing & Global Registration
    voxel_size = 0.05
    print(":: Performing Global Registration (RANSAC)...")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    global_result = execute_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh, voxel_size)
    source.transform(global_result.transformation)

    # 3. Fine Registration (Point-to-Plane ICP)
    print(":: Performing Fine Registration (ICP)...")
    target.estimate_normals()
    source.estimate_normals()
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    source.transform(icp_result.transformation)

    # 4. Heatmap Computation
    distances = source.compute_point_cloud_distance(target)
    dist_array = np.asarray(distances)

    # Dynamic coloring based on mean distance
    max_dist = dist_array.mean() * 2
    colors = plt.get_cmap("jet")(dist_array / max_dist)[:, :3]
    source.colors = o3d.utility.Vector3dVector(colors)

    print(f":: Final Mean Distance: {np.mean(dist_array):.6f}")
    o3d.visualization.draw_geometries([source], window_name="Aligned Heatmap Result")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Point Cloud to Mesh/Cloud Comparison Tool")
    parser.add_argument("--source", type=str, help="Path to source file")
    parser.add_argument("--target", type=str, help="Path to target file")
    parser.add_argument("--is_mesh", action="store_true", help="Flag if target is a Mesh")
    args = parser.parse_args()

    if args.source and args.target:
        run_comparison(args.source, args.target, args.is_mesh)
    else:
        print(":: No paths provided. Running official Demo data...")
        knot = o3d.data.KnotMesh()
        demo_pcd = o3d.data.DemoICPPointClouds()
        run_comparison(demo_pcd.paths[0], knot.path, is_target_mesh=True)
        