# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys
import urllib.request
import zipfile

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
    import open3d as o3d
    if o3d.core.cuda.device_count() > 0:
        return [o3d.core.Device("CPU:0"), o3d.core.Device("CUDA:0")]
    else:
        return [o3d.core.Device("CPU:0")]


def list_devices_with_torch():
    """
    Similar to list_devices(), but take PyTorch available devices into account.
    The returned devices are compatible on both PyTorch and Open3D.

    If PyTorch is not available at all, empty list will be returned, thus the
    test is effectively skipped.
    """
    if torch_available():
        import open3d as o3d
        import torch
        if (o3d.core.cuda.device_count() > 0 and torch.cuda.is_available() and
                torch.cuda.device_count() > 0):
            return [o3d.core.Device("CPU:0"), o3d.core.Device("CUDA:0")]
        else:
            return [o3d.core.Device("CPU:0")]
    else:
        return []
