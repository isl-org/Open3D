# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


class Dog():

    def __init__(self):
        pass

    def __getitem__(self, key):
        print("__getitem__")
        print(key)


def test_dog():
    pass
    # print("dog")
    # d = Dog()
    # d[(1, 2, 3)]
    # d[1, 2:3, [1, 2, 3]]
    # print("dog done")


def test_creation():
    t = o3d.pybind.core.Tensor.empty((20, 30))
    t = o3d.pybind.core.Tensor.empty([20, 30])


# def test_tuple():
#     t = o3d.pybind.core.Tensor(np.ones((30, 40, 50)))
#     t_idx = o3d.pybind.core.Tensor([3, 4, 5])
#     t[t_idx, 1:3, 5]
#     t[1:3, 5, t]
#     t[[3, 4, 5], 1:3, 5]
#     t[(3, 4, 5), 1:3, 5]
#     t[np.array([3, 4, 5]), 1:3, 5]

# def test_getitem():
#     t = o3d.pybind.core.Tensor([0, 1, 2, 3, 4, 5],
#                                dtype=None,
#                                device=o3c.Device("CPU:0"))
#     t[1]
#     t[1:3]
#     t[[1, 3, 4]]
#     t[[True, False, False, True, True, True]]
#     t[np.array([1, 3, 4])]
#     t[np.array([True, False, False, True, True, True])]
#     t[o3d.pybind.core.Tensor([1, 3, 4])]
#     t[o3d.pybind.core.Tensor([True, False, False, True, True, True])]

# def test_setitem():
#     t = o3d.pybind.core.Tensor([0, 1, 2, 3, 4, 5],
#                                dtype=None,
#                                device=o3c.Device("CPU:0"))
#     t[1] = 100
#     t[1:3] = o3d.pybind.core.Tensor([4, 5],
#                                     dtype=None,
#                                     device=o3c.Device("CPU:0"))
#     t[[1, 3, 4]] = 100
#     t[[True, False, False, True, True, True]] = 100
#     t[np.array([1, 3, 4])] = 100
#     t[np.array([True, False, False, True, True, True])] = 100
#     t[o3d.pybind.core.Tensor([1, 3, 4])] = 100
#     t[o3d.pybind.core.Tensor([True, False, False, True, True, True])] = 100

# def test_creation():
#     t_np = np.ones((2, 3))

#     t0 = o3c.Tensor(t_np)
#     # print(t0)

#     t1 = o3d.pybind.core.Tensor(t_np, o3c.Dtype.Int32)
#     # print(t1)

#     t1 = o3d.pybind.core.Tensor(t_np, dtype=None, device=o3c.Device("CPU:0"))
#     # print(t1)

#     t1 = o3d.pybind.core.Tensor(1,
#                                 dtype=o3c.Dtype.Float32,
#                                 device=o3c.Device("CPU:0"))
#     # print(t1)

#     t1 = o3d.pybind.core.Tensor(3.14,
#                                 dtype=o3c.Dtype.Int32,
#                                 device=o3c.Device("CPU:0"))
#     # print(t1)

# def test_list_creation():
#     t1 = o3d.pybind.core.Tensor([0, 1, 2, 3, 4, 5],
#                                 dtype=None,
#                                 device=o3c.Device("CPU:0"))

#     t1 = o3d.pybind.core.Tensor([[0, 1, 2], [3, 4, 5]],
#                                 dtype=None,
#                                 device=o3c.Device("CPU:0"))

#     with pytest.raises(RuntimeError):
#         t1 = o3d.pybind.core.Tensor([[0, 1, 2, 3, 4, 5], [2, 3]],
#                                     dtype=None,
#                                     device=o3c.Device("CPU:0"))

# def test_tuple_creation():
#     t1 = o3d.pybind.core.Tensor((0, 1, 2, 3, 4, 5),
#                                 dtype=None,
#                                 device=o3c.Device("CPU:0"))

#     with pytest.raises(RuntimeError):
#         t1 = o3d.pybind.core.Tensor(((0, 1, 2, 3, 4, 5), (2, 3)),
#                                     dtype=None,
#                                     device=o3c.Device("CPU:0"))
