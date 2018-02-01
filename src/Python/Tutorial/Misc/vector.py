# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import copy
import sys
import numpy as np
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	print("Testing vector in py3d ...")

	print("")
	print("Testing IntVector ...")
	vi = IntVector([1, 2, 3, 4, 5]) # made from python list
	vi1 = IntVector(np.asarray([1, 2, 3, 4, 5])) # made from numpy array
	vi2 = copy.copy(vi) # valid copy
	vi3 = copy.deepcopy(vi) # valid copy
	vi4 = vi[:] # valid copy
	print(vi)
	print(np.asarray(vi))
	vi[0] = 10
	np.asarray(vi)[1] = 22
	vi1[0] *= 5
	vi2[0] += 1
	vi3[0:2] = IntVector([40, 50])
	print(vi)
	print(vi1)
	print(vi2)
	print(vi3)
	print(vi4)

	print("")
	print("Testing DoubleVector ...")
	vd = DoubleVector([1, 2, 3])
	vd1 = DoubleVector([1.1, 1.2])
	vd2 = DoubleVector(np.asarray([0.1, 0.2]))
	print(vd)
	print(vd1)
	print(vd2)
	vd1.append(1.3)
	vd1.extend(vd2)
	print(vd1)

	print("")
	print("Testing Vector3dVector ...")
	vv3d = Vector3dVector([[1, 2, 3], [0.1, 0.2, 0.3]])
	vv3d1 = Vector3dVector(vv3d)
	vv3d2 = Vector3dVector(np.asarray(vv3d))
	vv3d3 = copy.deepcopy(vv3d)
	print(vv3d)
	print(np.asarray(vv3d))
	vv3d[0] = [4, 5, 6]
	print(np.asarray(vv3d))
	# bad practice, the second [] will not support slice
	vv3d[0][0] = -1
	print(np.asarray(vv3d))
	# good practice, use [] after converting to numpy.array
	np.asarray(vv3d)[0][0] = 0
	print(np.asarray(vv3d))
	np.asarray(vv3d1)[:2, :2] = [[10, 11], [12, 13]]
	print(np.asarray(vv3d1))
	vv3d2.append([30, 31, 32])
	print(np.asarray(vv3d2))
	vv3d3.extend(vv3d)
	print(np.asarray(vv3d3))

	print("")
	print("Testing Vector3iVector ...")
	vv3i = Vector3iVector([[1, 2, 3], [4, 5, 6]])
	print(vv3i)
	print(np.asarray(vv3i))

	print("")
