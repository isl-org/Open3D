# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import pytest
import open3d as o3d

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_bev_iou(device):
    # (x_center, z_center, x_size, z_size, y_rotate)
    boxes_0 = np.array(
        [[2.72802408, 7.45882562, 0.66284931, 0.50729007, 3.10826707],
         [-12.76986872, 19.57236453, 3.99570131, 1.61373711, -2.76809883],
         [-2.58648862, 31.65405458, 4.07396841, 1.58861041, -1.8442196],
         [-12.69745747, 38.14453306, 3.8139441, 1.5743469, 1.8430678],
         [-24.01241376, 32.02714812, 4.27763176, 1.69929349, -2.42430088],
         [-2.48355601, 48.38333119, 3.66774344, 1.5648849, -1.68973509],
         [-1.07197606, 65.3773688, 3.7635324, 1.65267885, 1.58406118]],
        dtype=np.float32)

    boxes_1 = np.array([[-2.61000003, 31.72999998, 4.32, 1.64, -1.84159258],
                        [-12.54000039, 19.71999952, 3.88, 1.62, -2.72159267],
                        [-12.66000011, 38.43999833, 4.29, 1.67, 1.72999993]],
                       dtype=np.float32)

    ref = np.array(
        [[0., 0., 0.], [0., 0.7067936, 0.], [0.9134971, 0., 0.],
         [0., 0., 0.7598292], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        dtype=np.float32)

    if device.get_type() == o3d.core.Device.DeviceType.CPU:
        from open3d.ml.contrib import iou_bev_cpu
        iou_bev = iou_bev_cpu
    elif device.get_type() == o3d.core.Device.DeviceType.CUDA:
        from open3d.ml.contrib import iou_bev_cuda
        iou_bev = iou_bev_cuda
    else:
        raise ("Unsupported device.")

    result = iou_bev(boxes_0, boxes_1)
    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-8)
    assert result.dtype == ref.dtype


@pytest.mark.parametrize("device", list_devices())
def test_3d_iou(device):
    # yapf: disable
    # (x_center, y_max, z_center, x_size, y_size, z_size, y_rotate)
    boxes_0 = np.array([[2.72802408, 0.91428565, 7.45882562, 0.66284931, 1.7254113, 0.50729007, 3.10826707],
                        [-12.76986872, 1.4418739, 19.57236453, 3.99570131, 1.4493804, 1.61373711, -2.76809883],
                        [-2.58648862, 1.01205023, 31.65405458, 4.07396841, 1.43224728, 1.58861041, -1.8442196],
                        [-12.69745747, 1.37015877, 38.14453306, 3.8139441, 1.43979049, 1.5743469, 1.8430678],
                        [-24.01241376, 0.97207756, 32.02714812, 4.27763176, 1.61623788, 1.69929349, -2.42430088],
                        [-2.48355601, 0.67658259, 48.38333119, 3.66774344, 1.51509488, 1.5648849, -1.68973509],
                        [-1.07197606, 0.48189101, 65.3773688, 3.7635324, 1.65629184, 1.65267885, 1.58406118]],
                        dtype=np.float32)

    boxes_1 = np.array([[-2.61000003, 1.12999999, 31.72999998, 4.32, 1.67, 1.64, -1.84159258],
                        [-12.54000039, 1.64000001, 19.71999952, 3.88, 1.5, 1.62, -2.72159267],
                        [-12.66000011, 1.13000002, 38.43999833, 4.29, 1.68, 1.67, 1.72999993]],
                        dtype=np.float32)

    ref = np.array([[0., 0., 0.],
                    [0., 0.57643616, 0.],
                    [0.7834455, 0., 0.],
                    [0., 0., 0.49211255],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]], dtype=np.float32)
    # yapf: enable

    if device.get_type() == o3d.core.Device.DeviceType.CPU:
        from open3d.ml.contrib import iou_3d_cpu
        iou_3d = iou_3d_cpu
    elif device.get_type() == o3d.core.Device.DeviceType.CUDA:
        from open3d.ml.contrib import iou_3d_cuda
        iou_3d = iou_3d_cuda
    else:
        raise ("Unsupported device.")

    result = iou_3d(boxes_0, boxes_1)
    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-8)
    assert result.dtype == ref.dtype
