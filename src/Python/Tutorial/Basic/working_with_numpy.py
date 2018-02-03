# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys, copy
import numpy as np
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	# generate some neat n times 3 matrix using a variant of sync function
	x = np.linspace(-3, 3, 401)
	mesh_x, mesh_y = np.meshgrid(x,x)
	z = np.sinc((np.power(mesh_x,2)+np.power(mesh_y,2)))
	xyz = np.zeros((np.size(mesh_x),3))
	xyz[:,0] = np.reshape(mesh_x,-1)
	xyz[:,1] = np.reshape(mesh_y,-1)
	xyz[:,2] = np.reshape(z,-1)
	print('xyz')
	print(xyz)

	# Pass xyz to Open3D.PointCloud and visualize
	pcd = PointCloud()
	pcd.points = Vector3dVector(xyz)
	write_point_cloud("../../TestData/sync.ply", pcd)

	# Load saved point cloud and transform it into numpy array
	pcd_load = read_point_cloud("../../TestData/sync.ply")
	xyz_load = np.asarray(pcd_load.points)
	print('xyz_load')
	print(xyz_load)

    # visualization
    draw_geometries([pcd_load])
