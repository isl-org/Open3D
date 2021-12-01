import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_buffer_protocol_cpu(device):
    if device.get_type() == o3c.Device.DeviceType.CPU:
        # (rows, cols) -> (rows, cols, 1)
        src_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        dst_t = np.asarray(im)
        np.testing.assert_array_equal(src_t[..., None], dst_t)

        # Check that the memory is shared.
        dst_t[0, 0, 0] = 100
        new_dst_t = np.asarray(im)
        np.testing.assert_array_equal(dst_t, new_dst_t)

        # (rows, cols, channels) -> (rows, cols, channels)
        src_t = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        dst_t = np.asarray(im)
        np.testing.assert_array_equal(src_t, dst_t)

        # Check that the memory is shared.
        dst_t[0, 0, 0] = 100
        new_dst_t = np.asarray(im)
        np.testing.assert_array_equal(dst_t, new_dst_t)
    else:
        # (rows, cols) -> (rows, cols, 1)
        src_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        im = im.to(device=device)
        # Ideally we shall test excpetion if .cpu() is not called, but
        # pytest.raises() cannot catch this exception for some reason.
        dst_t = np.asarray(im.cpu())
        np.testing.assert_array_equal(src_t[..., None], dst_t)

        # (rows, cols, channels) -> (rows, cols, channels)
        src_t = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        im = im.to(device=device)
        dst_t = np.asarray(im.cpu())
        np.testing.assert_array_equal(src_t, dst_t)
