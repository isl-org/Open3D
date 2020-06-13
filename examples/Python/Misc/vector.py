# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Misc/vector.py

import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Testing vector in open3d ...")

    print("")
    print("Testing o3d.utility.IntVector ...")
    vi = o3d.utility.IntVector([1, 2, 3, 4, 5])  # made from python list
    vi1 = o3d.utility.IntVector(np.asarray([1, 2, 3, 4,
                                            5]))  # made from numpy array
    vi2 = copy.copy(vi)  # valid copy
    vi3 = copy.deepcopy(vi)  # valid copy
    vi4 = vi[:]  # valid copy
    print(vi)
    print(np.asarray(vi))
    vi[0] = 10
    np.asarray(vi)[1] = 22
    vi1[0] *= 5
    vi2[0] += 1
    vi3[0:2] = o3d.utility.IntVector([40, 50])
    print(vi)
    print(vi1)
    print(vi2)
    print(vi3)
    print(vi4)

    print("")
    print("Testing o3d.utility.DoubleVector ...")
    vd = o3d.utility.DoubleVector([1, 2, 3])
    vd1 = o3d.utility.DoubleVector([1.1, 1.2])
    vd2 = o3d.utility.DoubleVector(np.asarray([0.1, 0.2]))
    print(vd)
    print(vd1)
    print(vd2)
    vd1.append(1.3)
    vd1.extend(vd2)
    print(vd1)

    print("")
    print("Testing o3d.utility.Vector3dVector ...")
    vv3d = o3d.utility.Vector3dVector([[1, 2, 3], [0.1, 0.2, 0.3]])
    vv3d1 = o3d.utility.Vector3dVector(vv3d)
    vv3d2 = o3d.utility.Vector3dVector(np.asarray(vv3d))
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
    print("Testing o3d.utility.Vector3iVector ...")
    vv3i = o3d.utility.Vector3iVector([[1, 2, 3], [4, 5, 6]])
    print(vv3i)
    print(np.asarray(vv3i))

    print("")
