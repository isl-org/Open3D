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
    from open3d.cuda.pybind.core import (Tensor, Dtype, DtypeCode, Device, cuda,
                                         nns, NoneType, TensorList, SizeVector,
                                         DynamicSizeVector, matmul, lstsq,
                                         solve, inv, svd)
else:
    from open3d.cpu.pybind.core import (Tensor, Dtype, DtypeCode, Device, cuda,
                                        nns, NoneType, TensorList, SizeVector,
                                        DynamicSizeVector, matmul, lstsq, solve,
                                        inv, svd)

none = NoneType()


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
