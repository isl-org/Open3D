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
import numpy as np
import pytest

import open3d as o3d
import numpy as np
import pytest


def torch_available():
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        return False
    return True


def list_devices():
    """
    If Open3D is built with CUDA support:
    - If cuda device is available, returns [Device("CPU:0"), Device("CUDA:0")].
    - If cuda device is not available, returns [Device("CPU:0")].

    If Open3D is built without CUDA support:
    - returns [Device("CPU:0")].
    """
    devices = [o3d.core.Device("CPU:" + str(0))]
    if torch_available() and o3d._build_config['BUILD_CUDA_MODULE']:
        import torch
        import torch.utils.dlpack
        if (o3d.core.cuda.device_count() != torch.cuda.device_count()):
            raise RuntimeError(
                "o3d.core.cuda.device_count() != torch.cuda.device_count(), "
                "{} != {}".format(o3d.core.cuda.device_count(),
                                  torch.cuda.device_count()))
    if o3d.core.cuda.device_count() > 0:
        devices.append(o3d.core.Device("CUDA:0"))
    return devices
