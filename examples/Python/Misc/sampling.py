# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Misc/sampling.py

import open3d as o3d
import os, sys
sys.path.append("../Utility")
from common import *
sys.path.append("../Advanced")
from trajectory_io import *
from shutil import copyfile

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    path = "[path_to_reconstruction_system_output]"
    out_path = "[path_to_sampled_frames_are_located]"
    make_clean_folder(out_path)
    make_clean_folder(os.path.join(out_path, "depth/"))
    make_clean_folder(os.path.join(out_path, "image/"))
    make_clean_folder(os.path.join(out_path, "scene/"))
    sampling_rate = 30

    depth_image_path = get_file_list(os.path.join(path, "depth/"),
                                     extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"),
                                     extension=".jpg")
    pose_graph_global = o3d.io.read_pose_graph(
        os.path.join(path, template_global_posegraph_optimized))
    n_fragments = len(depth_image_path) // n_frames_per_fragment + 1
    pose_graph_fragments = []
    for i in range(n_fragments):
        pose_graph_fragment = o3d.io.read_pose_graph(
            os.path.join(path, template_fragment_posegraph_optimized % i))
        pose_graph_fragments.append(pose_graph_fragment)

    depth_image_path_new = []
    color_image_path_new = []
    traj = []
    cnt = 0
    for i in range(len(depth_image_path)):
        if i % sampling_rate == 0:
            metadata = [cnt, cnt, len(depth_image_path) // sampling_rate + 1]
            print(metadata)
            fragment_id = i // n_frames_per_fragment
            local_frame_id = i - fragment_id * n_frames_per_fragment
            traj.append(
                CameraPose(
                    metadata,
                    np.dot(
                        pose_graph_global.nodes[fragment_id].pose,
                        pose_graph_fragments[fragment_id].nodes[local_frame_id].
                        pose)))
            copyfile(depth_image_path[i], os.path.join(out_path, "depth/", \
                    os.path.basename(depth_image_path[i])))
            copyfile(color_image_path[i], os.path.join(out_path, "image/", \
                    os.path.basename(color_image_path[i])))
            cnt += 1
    copyfile(os.path.join(path, "/scene/cropped.ply"),
             os.path.join(out_path, "/scene/integrated.ply"))
    write_trajectory(traj, os.path.join(out_path, "scene/key.log"))
