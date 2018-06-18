# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import copy
from open3d import *
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

folder_fragment = "/fragments/"
template_fragment_posegraph = folder_fragment + "fragment_%03d.json"
template_fragment_posegraph_optimized = folder_fragment + \
        "fragment_optimized_%03d.json"
template_fragment_mesh = folder_fragment + "fragment_%03d.ply"
folder_scene = "/scene/"
template_global_posegraph = folder_scene + "global_registration.json"
template_global_posegraph_optimized = folder_scene + \
        "global_registration_optimized.json"
template_global_mesh = folder_scene + "integrated.ply"


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [path + f for f in listdir(path)
                if isfile(join(path, f)) and splitext(f)[1] == extension]
    file_list.sort()
    return file_list


def get_rgbd_file_lists(path_dataset):
    path_color = path_dataset + "/image/"
    path_depth = path_dataset + "/depth/"
    color_files = get_file_list(path_color, ".jpg") + \
            get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)


#######################
# visualization related
#######################
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)
    draw_geometries([source_temp, target_temp])


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)
    draw_geometries([source_temp, target_temp])
