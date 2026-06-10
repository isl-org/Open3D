# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os


def torch_available():
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        return False
    return True


def _use_sycl_cpu_fallback_for_ci(num_sycl_devices,
                                  use_sycl_cpu_fallback_in_ci):
    return (use_sycl_cpu_fallback_in_ci and os.getenv("CI") is not None and
            num_sycl_devices == 1)


def list_devices(enable_cuda=True,
                 enable_sycl=False,
                 use_sycl_cpu_fallback_in_ci=True):
    """
    Returns a list of devices that are available for Open3D to use:
    - Device("CPU:0")
    - Device("CUDA:0") if built with CUDA support and a CUDA device is available.
    - Device("SYCL:0") if built with SYCL support and a SYCL GPU device is available.
    - Device("SYCL:0") in CI when the SYCL CPU fallback is the only SYCL device,
      unless the caller disables that fallback for hardware-specific tests.
    """
    import open3d as o3d

    devices = [o3d.core.Device("CPU:0")]
    if enable_cuda and o3d.core.cuda.device_count() > 0:
        devices.append(o3d.core.Device("CUDA:0"))
    num_sycl_devices = len(o3d.core.sycl.get_available_devices())
    if enable_sycl and (num_sycl_devices > 1 or _use_sycl_cpu_fallback_for_ci(
            num_sycl_devices, use_sycl_cpu_fallback_in_ci)):
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
        devices = [o3d.core.Device("CPU:0")]
        if (o3d.core.cuda.device_count() > 0 and torch.cuda.device_count() > 0):
            devices += [o3d.core.Device("CUDA:0")]
        # Last SYCL device is CPU, so there must be 2+ devices in Open3D here.
        if (o3d.core.sycl.device_count() > 1 and torch.xpu.device_count() > 0):
            devices += [o3d.core.Device("SYCL:0")]
        return devices
    else:
        return []
