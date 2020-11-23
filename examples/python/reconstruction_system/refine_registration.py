# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/refine_registration.py

import numpy as np
import open3d as o3d
import sys
sys.path.append("../utility")
from file import join, get_file_list
from visualization import draw_registration_result_original_color
sys.path.append(".")
from optimize_posegraph import optimize_posegraph_for_refined_scene


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


def multiscale_icp(source,
                   target,
                   voxel_size,
                   max_iter,
                   config,
                   init_transformation=np.identity(4)):
    current_transformation = init_transformation
    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = config["voxel_size"] * 1.4
        print("voxel_size {}".format(voxel_size[scale]))
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])
        if config["icp_method"] == "point_to_point":
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iter))
        else:
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            if config["icp_method"] == "point_to_plane":
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=iter))
            if config["icp_method"] == "color":
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, voxel_size[scale],
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=iter))
        current_transformation = result_icp.transformation
        if i == len(max_iter) - 1:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, voxel_size[scale] * 1.4,
                result_icp.transformation)

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target,
                                                result_icp.transformation)
    return (result_icp.transformation, information_matrix)


def local_refinement(source, target, transformation_init, config):
    voxel_size = config["voxel_size"]
    (transformation, information) = \
            multiscale_icp(
            source, target,
            [voxel_size, voxel_size/2.0, voxel_size/4.0], [50, 30, 14],
            config, transformation_init)
    if config["debug_mode"]:
        draw_registration_result_original_color(source, target, transformation)
    return (transformation, information)


def register_point_cloud_pair(ply_file_names, s, t, transformation_init,
                              config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])
    (transformation, information) = \
            local_refinement(source, target, transformation_init, config)
    if config["debug_mode"]:
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
        matching_results[s * n_files + t] = \
                matching_result(s, t, edge.transformation)

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
