# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
from os import listdir
from os.path import isfile, join, splitext

#######################
# some global parameters for the global registration
#######################
n_frames_per_fragment = 100
n_keyframes_per_n_frame = 5



#######################
# file related
#######################

def get_file_list(path, extension=None):
	if extension is None:
		file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
	else:
		file_list = [path + f for f in listdir(path)
				if isfile(join(path, f)) and splitext(f)[1] == extension]
	file_list.sort()
	return file_list


def get_file_lists(path_dataset):
	# get list of color and depth images
	path_color = path_dataset + '/image/'
	path_depth = path_dataset + '/depth/'
	color_files = get_file_list(path_color, '.png')
	depth_files = get_file_list(path_depth, '.png')
	return color_files, depth_files


def get_file_list_from_custom_format(path, format):
	number_of_files = len(get_file_list(path, splitext(format)[1]))
	file_list = []
	for i in range(number_of_files):
		file_list.append("%s/%s" % (path, format % i))
	return file_list


#######################
# visualization related
#######################

def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.paint_uniform_color([1, 0.706, 0])
	target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)
	draw_geometries([source_temp, target_temp])


def draw_registration_result_original_color(source, target, transformation):
	source_temp = copy.deepcopy(source)
	source_temp.transform(transformation)
	draw_geometries([source_temp, target])
