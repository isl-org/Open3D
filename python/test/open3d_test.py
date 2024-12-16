# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------


def torch_available():
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        return False
    return True


def list_devices(enable_cuda=True, enable_sycl=False):
    """
    Returns a list of devices that are available for Open3D to use:
    - Device("CPU:0")
    - Device("CUDA:0") if built with CUDA support and a CUDA device is available.
    - Device("SYCL:0") if built with SYCL support and a SYCL GPU device is available.
    """
    import open3d as o3d

    devices = [o3d.core.Device("CPU:0")]
    if enable_cuda and o3d.core.cuda.device_count() > 0:
        devices.append(o3d.core.Device("CUDA:0"))
    # Ignore fallback SYCL CPU device
    if enable_sycl and len(o3d.core.sycl.get_available_devices()) > 1:
        devices.append(o3d.core.Device("SYCL:0"))
    return devices


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
