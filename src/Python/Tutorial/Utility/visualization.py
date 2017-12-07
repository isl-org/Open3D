# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys, copy
sys.path.append("../..")
from py3d import *


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
