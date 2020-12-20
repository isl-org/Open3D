# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
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
import numpy as np

if o3d.__DEVICE_API__ == 'cuda':
    from open3d.cuda.pybind.core import (Dtype, DtypeCode, Device, cuda, nns,
                                         NoneType, TensorList, SizeVector,
                                         DynamicSizeVector, matmul, lstsq,
                                         solve, inv, svd)
else:
    from open3d.cpu.pybind.core import (Dtype, DtypeCode, Device, cuda, nns,
                                        NoneType, TensorList, SizeVector,
                                        DynamicSizeVector, matmul, lstsq, solve,
                                        inv, svd)

none = NoneType()


def cast_to_py_tensor(func):
    """
    Args:
        func: function returning a `o3d.pybind.core.Tensor`.

    Return:
        A function which returns a python object `Tensor`.
    """

    def _maybe_to_py_tensor(c_tensor):
        if isinstance(c_tensor, o3d.pybind.core.Tensor):
            py_tensor = Tensor([])
            py_tensor.shallow_copy_from(c_tensor)
            return py_tensor
        else:
            return c_tensor

    def wrapped_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, list):
            return [_maybe_to_py_tensor(val) for val in result]
        elif isinstance(result, tuple):
            return tuple([_maybe_to_py_tensor(val) for val in result])
        else:
            return _maybe_to_py_tensor(result)

    return wrapped_func


# @cast_to_py_tensor
# def matmul(lhs, rhs):
#     """
#     Matrix multiplication between Tensor \param lhs and Tensor \param rhs

#     Args:
#       lhs: Tensor of shape (m, k)
#       rhs: Tensor of shape (k, n)

#     Returns:
#       Tensor of shape (m, n)

#     - Both tensors should share the same device and dtype.
#     - Int32, Int64, Float32, Float64 are supported,
#       but results of big integers' matmul are not guaranteed, overflow can
#       happen.
#     """
#     return pybind_matmul(lhs, rhs)

# @cast_to_py_tensor
# def solve(lhs, rhs):
#     """
#     Returns X by solving linear system AX = B with LU decomposition,
#     where A is Tensor \param lhs and B is Tensor \param rhs.

#     Args:
#       lhs: Tensor of shape (n, n)
#       rhs: Tensor of shape (n, k)

#     Returns:
#       Tensor of shape (n, k)

#     - Both tensors should share the same device and dtype.
#     - Float32 and Float64 are supported.
#     """
#     return pybind_solve(lhs, rhs)

# @cast_to_py_tensor
# def lstsq(lhs, rhs):
#     """
#     Returns X by solving linear system AX = B with QR decomposition,
#     where A is Tensor \param lhs and B is Tensor \param rhs.

#     Args:
#       lhs: Tensor of shape (m, n), m >= n and is a full rank matrix.
#       rhs: Tensor of shape (m, k)

#     Returns:
#       Tensor of shape (n, k)

#     - Both tensors should share the same device and dtype.
#     - Float32 and Float64 are supported.
#     """
#     return pybind_lstsq(lhs, rhs)

# @cast_to_py_tensor
# def inv(val):
#     """
#     Returns matrix's inversion with LU decomposition.

#     Args:
#       val: Tensor of shape (m, m) and is an invertable matrix

#     Returns:
#       Tensor of shape (m, m)

#     - Float32 and Float64 are supported.
#     """
#     return pybind_inv(val)

# @cast_to_py_tensor
# def svd(val):
#     """
#     Returns matrix's SVD decomposition: U S VT = A, where A is Tensor \param val.

#     Args:
#       val: Tensor of shape (m, n).

#     Returns: a tuple of tensors:
#       U: Tensor of shape (m, n)
#       S: Tensor of shape (min(m, n))
#       VT: Tensor of shape (n, n)

#     - Float32 and Float64 are supported.
#     """
#     return pybind_svd(val)


class Tensor(o3d.pybind.core.Tensor):
    """
    Open3D Tensor class. A Tensor is a view of data blob with shape, strides
    and etc. Tensor can be used to perform numerical operations.
    """

    pass


class Hashmap(o3d.pybind.core.Hashmap):
    """
    Open3D Hashmap class. A Hashmap is a map from key to data wrapped by Tensors.
    """

    def __init__(self,
                 init_capacity,
                 dtype_key,
                 dtype_value,
                 shape_key=[1],
                 shape_value=[1],
                 device=None):
        if not isinstance(shape_key, SizeVector):
            shape_key = SizeVector(shape_key)
        if not isinstance(shape_value, SizeVector):
            shape_value = SizeVector(shape_value)
        super(Hashmap, self).__init__(init_capacity, dtype_key, dtype_value,
                                      shape_key, shape_value, device)

    @cast_to_py_tensor
    def insert(self, keys, values):
        return super(Hashmap, self).insert(keys, values)

    @cast_to_py_tensor
    def find(self, keys):
        return super(Hashmap, self).find(keys)

    @cast_to_py_tensor
    def activate(self, keys):
        return super(Hashmap, self).activate(keys)

    @cast_to_py_tensor
    def erase(self, keys):
        return super(Hashmap, self).erase(keys)

    @cast_to_py_tensor
    def get_active_addrs(self):
        return super(Hashmap, self).get_active_addrs()

    @cast_to_py_tensor
    def get_key_buffer(self):
        return super(Hashmap, self).get_key_buffer()

    @cast_to_py_tensor
    def get_value_buffer(self):
        return super(Hashmap, self).get_value_buffer()

    @cast_to_py_tensor
    def get_key_tensor(self):
        return super(Hashmap, self).get_key_tensor()

    @cast_to_py_tensor
    def get_value_tensor(self):
        return super(Hashmap, self).get_value_tensor()
