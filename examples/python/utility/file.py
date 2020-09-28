# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/utility/file.py

from os import listdir, makedirs
from os.path import exists, isfile, join, splitext
import shutil
import re


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
    return path


def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth


def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + \
            get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        makedirs(path_folder)


def check_folder_structure(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    assert exists(path_depth), \
            "Path %s is not exist!" % path_depth
    assert exists(path_color), \
            "Path %s is not exist!" % path_color


def write_poses_to_log(filename, poses):
    with open(filename, 'w') as f:
        for i, pose in enumerate(poses):
            f.write('{} {} {}\n'.format(i, i, i + 1))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]))
