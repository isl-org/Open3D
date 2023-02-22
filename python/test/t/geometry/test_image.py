# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.core as o3c
from open3d.t.geometry import Image
import numpy as np
import pytest
import pickle
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_to_linear_transform(device):
    TOL = {"rtol": 1e-7, "atol": 1e-7}
    # reference data
    input_data = np.array((10, 25, 0, 13, 5, 40), dtype=np.uint8).reshape(
        (2, 3, 1))
    output_ref = input_data / 255.
    negative_image_ref = 1. - input_data / 255.
    saturate_ref = np.array((180, 255, 0, 240, 80, 255),
                            dtype=np.uint8).reshape((2, 3, 1))

    t_input = o3c.Tensor(input_data, dtype=o3c.uint8, device=device)
    t_input3 = o3c.Tensor(np.broadcast_to(input_data, shape=(2, 3, 3)),
                          dtype=o3c.uint8,
                          device=device)

    input1 = Image(t_input)
    # UInt8 -> Float32: auto scale = 1./255
    output1 = input1.to(o3c.float32)
    assert output1.dtype == o3c.float32
    np.testing.assert_allclose(output1.as_tensor().cpu().numpy(), output_ref,
                               **TOL)
    # 3 channels
    input3 = Image(t_input3)
    output3 = input3.to(o3c.float32)
    np.testing.assert_allclose(output3.as_tensor().cpu().numpy(),
                               np.broadcast_to(output_ref, (2, 3, 3)), **TOL)

    # LinearTransform to negative image
    output1.linear_transform(scale=-1, offset=1)
    np.testing.assert_allclose(output1.as_tensor().cpu().numpy(),
                               negative_image_ref)
    # 3 channels
    output3.linear_transform(scale=-1, offset=1)
    np.testing.assert_allclose(output3.as_tensor().cpu().numpy(),
                               np.broadcast_to(negative_image_ref, (2, 3, 3)),
                               **TOL)

    # UInt8 -> UInt16: auto scale = 1
    output1 = input1.to(o3c.uint16)
    assert output1.dtype == o3c.uint16
    np.testing.assert_allclose(output1.as_tensor().cpu().numpy(), input_data,
                               **TOL)
    # 3 channels
    output3 = input3.to(o3c.uint16)
    np.testing.assert_allclose(output3.as_tensor().cpu().numpy(),
                               np.broadcast_to(input_data, (2, 3, 3)), **TOL)

    # Saturation to [0, 255]
    output1 = input1.linear_transform(scale=20, offset=-20)
    np.testing.assert_allclose(output1.as_tensor().cpu().numpy(), saturate_ref,
                               **TOL)
    # 3 channels
    output3 = input3.linear_transform(scale=20, offset=-20)
    np.testing.assert_allclose(output3.as_tensor().cpu().numpy(),
                               np.broadcast_to(saturate_ref, (2, 3, 3)), **TOL)


@pytest.mark.parametrize("device", list_devices())
def test_buffer_protocol_cpu(device):
    if device.get_type() == o3c.Device.DeviceType.CPU:
        # (rows, cols) -> (rows, cols, 1)
        src_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        im = Image(o3d.core.Tensor.from_numpy(src_t))
        dst_t = np.asarray(im)
        np.testing.assert_array_equal(src_t[..., None], dst_t)

        # Check that the memory is shared.
        dst_t[0, 0, 0] = 100
        new_dst_t = np.asarray(im)
        np.testing.assert_array_equal(dst_t, new_dst_t)

        # (rows, cols, channels) -> (rows, cols, channels)
        src_t = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
        im = Image(o3d.core.Tensor.from_numpy(src_t))
        dst_t = np.asarray(im)
        np.testing.assert_array_equal(src_t, dst_t)

        # Check that the memory is shared.
        dst_t[0, 0, 0] = 100
        new_dst_t = np.asarray(im)
        np.testing.assert_array_equal(dst_t, new_dst_t)
    else:
        # (rows, cols) -> (rows, cols, 1)
        src_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        im = Image(o3d.core.Tensor.from_numpy(src_t))
        im = im.to(device=device)
        # Ideally we shall test exception if .cpu() is not called, but
        # pytest.raises() cannot catch this exception for some reason.
        dst_t = np.asarray(im.cpu())
        np.testing.assert_array_equal(src_t[..., None], dst_t)

        # (rows, cols, channels) -> (rows, cols, channels)
        src_t = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
        im = Image(o3d.core.Tensor.from_numpy(src_t))
        im = im.to(device=device)
        dst_t = np.asarray(im.cpu())
        np.testing.assert_array_equal(src_t, dst_t)


@pytest.mark.parametrize("device", list_devices())
def test_pickle(device):
    img = Image(o3c.Tensor.ones((10, 10, 3), o3c.uint8, device))
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/img.pkl"
        pickle.dump(img, open(file_name, "wb"))
        img_load = pickle.load(open(file_name, "rb"))
        assert img_load.as_tensor().allclose(img.as_tensor())
        assert img_load.device == img.device and img_load.dtype == o3c.uint8
