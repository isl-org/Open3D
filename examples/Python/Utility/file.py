# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import re
import os
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext

#######################
# some global parameters for the global registration
#######################

n_frames_per_fragment = 100
n_keyframes_per_n_frame = 5


#######################
# file related
#######################

folder_fragment = "fragments/"
template_fragment_posegraph = os.path.join(
        folder_fragment, "fragment_%03d.json")
template_fragment_posegraph_optimized = os.path.join(
        folder_fragment, "fragment_optimized_%03d.json")
template_fragment_mesh = os.path.join(folder_fragment, "fragment_%03d.ply")
folder_scene = "scene/"
template_global_posegraph = os.path.join(
        folder_scene, "global_registration.json")
template_global_posegraph_optimized = os.path.join(folder_scene,
        "global_registration_optimized.json")
template_refined_posegraph = os.path.join(
        folder_scene, "refined_registration.json")
template_refined_posegraph_optimized = os.path.join(folder_scene,
        "refined_registration_optimized.json")
template_global_mesh = os.path.join(folder_scene, "integrated.ply")


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [path + f for f in listdir(path)
                if isfile(join(path, f)) and splitext(f)[1] == extension]
    file_list = sorted_alphanum(file_list)
    return file_list


def get_rgbd_file_lists(path_dataset):
    if os.path.exists(os.path.join(path_dataset, "image/")):
        path_color = os.path.join(path_dataset, "image/")
    else:
        path_color = os.path.join(path_dataset, "rgb/")
    path_depth = os.path.join(path_dataset, "depth/")
    color_files = get_file_list(path_color, ".jpg") + \
            get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
