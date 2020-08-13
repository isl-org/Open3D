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
import pytest

try:
    import torch
    import torch.utils.dlpack
except ImportError:
    _torch_imported = False
else:
    _torch_imported = True


def list_devices():
    """
    If Open3D is built with CUDA support:
    - If cuda device is available, returns [Device("CPU:0"), Device("CUDA:0")].
    - If cuda device is not available, returns [Device("CPU:0")].

    If Open3D is built without CUDA support:
    - returns [Device("CPU:0")].
    """
    devices = [o3d.core.Device("CPU:" + str(0))]
    if _torch_imported and o3d._build_config['BUILD_CUDA_MODULE']:
        if (o3d.core.cuda.device_count() != torch.cuda.device_count()):
            raise RuntimeError(
                "o3d.core.cuda.device_count() != torch.cuda.device_count(), "
                "{} != {}".format(o3d.core.cuda.device_count(),
                                  torch.cuda.device_count()))
    else:
        print("Warning: PyTorch is not imported")
    if o3d.core.cuda.device_count() > 0:
        devices.append(o3d.core.Device("CUDA:0"))
    return devices


@pytest.mark.parametrize("device", list_devices())
def test_tensor_constructor(device):
    print(device)
    dtype = o3d.core.Dtype.Int32

    # Numpy array
    np_t = np.random.rand(100, 3).astype(np.float32)
    o3_t = o3d.core.Tensor(np_t, dtype, device)
    index = o3d.geometry.KnnFaiss()
    index.set_tensor_data(o3_t)
    [_, indices, _] = index.search_knn_vector_3d(np_t[0], knn=1)
    np.testing.assert_equal(np.asarray(indices), np.array([0]))
