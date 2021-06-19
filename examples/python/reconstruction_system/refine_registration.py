# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/refine_registration.py

from optimize_posegraph import optimize_posegraph_for_refined_scene
from visualization import draw_registration_result_original_color
from file import join, get_file_list, write_poses_to_log
import numpy as np
import open3d as o3d
import sys
sys.path.append("../utility")
sys.path.append(".")


def update_posegraph_for_scene(s, t, transformation, information, odometry,
                               pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=True))
    return (odometry, pose_graph)


def local_refinement(source, target, transformation_init, config):
    voxel_size = config["voxel_size"]

    voxel_radius = o3d.utility.DoubleVector(
        [voxel_size, voxel_size / 2.0, voxel_size / 4.0])
    distance_threshold = o3d.utility.DoubleVector(
        [voxel_size * 1.4, voxel_size * 1.4 / 2.0, voxel_size * 1.4 / 4.0])

    criteria_list = [
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-3,
                                                          relative_rmse=1e-3,
                                                          max_iteration=50),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-4,
                                                          relative_rmse=1e-4,
                                                          max_iteration=30),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=14)
    ]

    result_icp = o3d.pipelines.registration.registration_result()

    if config["icp_method"] == "point_to_point":
        result_icp = o3d.pipelines.registration.registration_multi_scale_icp(
            source, target, voxel_radius, criteria_list, distance_threshold,
            transformation_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius[2] * 2.0,
                                                 max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius[2] * 2.0,
                                                 max_nn=30))

        if config["icp_method"] == "point_to_plane":
            result_icp = o3d.pipelines.registration.registration_multi_scale_icp(
                source, target, voxel_radius, criteria_list, distance_threshold,
                transformation_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                ))
        elif config["icp_method"] == "color":
            result_icp = o3d.pipelines.registration.registration_multi_scale_icp(
                source, target, voxel_radius, criteria_list, distance_threshold,
                transformation_init,
                o3d.pipelines.registration.
                TransformationEstimationForColoredICP())

    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source.voxel_down_sample(voxel_radius[2]),
        target.voxel_down_sample(voxel_radius[2]), voxel_radius[2] * 1.4,
        result_icp.transformation)

    return (result_icp.transformation, information_matrix)


def register_point_cloud_pair(ply_file_names, s, t, transformation_init,
                              config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target,
                                                transformation_init)

    (transformation, information) = \
        local_refinement(source, target, transformation_init, config)

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target, transformation)
        print(transformation)
        print(information)
    return (transformation, information)


# other types instead of class?
class matching_result:

    def __init__(self, s, t, trans):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans
        self.infomation = np.identity(6)


def make_posegraph_for_refined_scene(ply_file_names, config):
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_optimized"]))

    n_files = len(ply_file_names)
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id

        transformation_init = edge.transformation
        matching_results[s * n_files + t] = \
            matching_result(s, t, transformation_init)

    if config["python_multi_threading"] == True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(pose_graph.edges), 1))
        results = Parallel(n_jobs=MAX_THREAD)(
            delayed(register_point_cloud_pair)(
                ply_file_names, matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation, config)
            for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].transformation = results[i][0]
            matching_results[r].information = results[i][1]
    else:
        for r in matching_results:
            (matching_results[r].transformation,
             matching_results[r].information) = \
                register_point_cloud_pair(ply_file_names,
                                          matching_results[r].s, matching_results[r].t,
                                          matching_results[r].transformation, config)

    pose_graph_new = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph_new.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(odometry))
    for r in matching_results:
        (odometry, pose_graph_new) = update_posegraph_for_scene(
            matching_results[r].s, matching_results[r].t,
            matching_results[r].transformation, matching_results[r].information,
            odometry, pose_graph_new)
    print(pose_graph_new)
    o3d.io.write_pose_graph(
        join(config["path_dataset"], config["template_refined_posegraph"]),
        pose_graph_new)


def run(config):
    print("refine rough registration of fragments.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")
    make_posegraph_for_refined_scene(ply_file_names, config)
    optimize_posegraph_for_refined_scene(config["path_dataset"], config)

    path_dataset = config['path_dataset']
    n_fragments = len(ply_file_names)

    # Save to trajectory
    poses = []
    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))
    for fragment_id in range(len(pose_graph_fragment.nodes)):
        pose_graph_rgbd = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % fragment_id))
        for frame_id in range(len(pose_graph_rgbd.nodes)):
            frame_id_abs = fragment_id * \
                config['n_frames_per_fragment'] + frame_id
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)
            poses.append(pose)

    traj_name = join(path_dataset, config["template_global_traj"])
    write_poses_to_log(traj_name, poses)
