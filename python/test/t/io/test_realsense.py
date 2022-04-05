# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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

# flake8: noqa: S101

import sys
import os
import shutil
import open3d as o3d
import numpy as np
import pytest


@pytest.mark.xfail(strict=False, reason="May fail depending on test state.")
@pytest.mark.skipif(os.getenv('GITHUB_SHA') is not None or
                    not hasattr(o3d.t.io, 'RSBagReader'),
                    reason="Hangs in Github Actions, succeeds locally or "
                    "not built with librealsense")
def test_RSBagReader():
    sample_l515_bag = o3d.data.SampleL515Bag()
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open(sample_l515_bag.path)

    # Metadata
    metadata = bag_reader.metadata
    assert metadata.color_channels == 3
    assert metadata.color_dt == o3d.core.uint8
    assert metadata.color_format == 'RGB8'
    assert metadata.depth_dt == o3d.core.uint16
    assert metadata.depth_format == 'Z16'
    assert np.allclose(metadata.depth_scale, 3999.999755859375)
    assert metadata.device_name == "Intel RealSense L515"
    assert metadata.fps == 30
    assert metadata.height == 540
    assert metadata.width == 960
    assert metadata.stream_length_usec == 199868
    assert np.allclose(
        metadata.intrinsics.intrinsic_matrix,
        np.array([[689.3069458, 0., 491.23974609],
                  [0., 689.74578857, 269.99111938], [0., 0., 1.]]))

    # Frames
    im_rgbd = bag_reader.next_frame()
    assert not im_rgbd.is_empty() and im_rgbd.are_aligned()
    assert im_rgbd.color.channels == 3
    assert im_rgbd.color.dtype == o3d.core.uint8
    assert im_rgbd.color.rows == 540
    assert im_rgbd.color.columns == 960
    assert im_rgbd.depth.channels == 1
    assert im_rgbd.depth.dtype == o3d.core.uint16
    assert im_rgbd.depth.rows == 540
    assert im_rgbd.depth.columns == 960

    n_frames = 0
    while not bag_reader.is_eof():
        n_frames = n_frames + 1
        im_rgbd = bag_reader.next_frame()

    bag_reader.close()
    assert n_frames == 6

    # save_frames
    bag_reader = o3d.t.io.RGBDVideoReader.create("L515_test_s.bag")
    bag_reader.save_frames("L515_test_s")
    # Use issubset() since there may be other OS files present
    assert {'depth', 'color',
            'intrinsic.json'}.issubset(os.listdir('L515_test_s'))
    assert {
        '00004.png', '00005.png', '00002.png', '00003.png', '00001.png',
        '00000.png'
    }.issubset(os.listdir('L515_test_s/depth'))
    assert {
        '00004.jpg', '00005.jpg', '00002.jpg', '00003.jpg', '00001.jpg',
        '00000.jpg'
    }.issubset(os.listdir('L515_test_s/color'))

    shutil.rmtree("L515_test_s")
    if os.name != 'nt':  # Permission error in Windows
        os.remove("L515_test_s.bag")


# Test recording from a RealSense camera, if one is connected
@pytest.mark.skipif(not hasattr(o3d.t.io, 'RealSenseSensor'),
                    reason="Not built with librealsense")
def test_RealSenseSensor():

    o3d.t.io.RealSenseSensor.list_devices()
    rs_cam = o3d.t.io.RealSenseSensor()
    bag_filename = "test_record.bag"
    try:
        rs_cam.init_sensor(o3d.t.io.RealSenseSensorConfig(), 0, bag_filename)
        rs_cam.start_capture(True)  # true: start recording with capture
        im_rgbd = rs_cam.capture_frame(True,
                                       True)  # wait for frames and align them
        assert im_rgbd.depth.rows == im_rgbd.color.rows > 0
        assert im_rgbd.depth.columns == im_rgbd.color.columns > 0
        rs_cam.stop_capture()
        assert os.path.exists(bag_filename)
        os.remove(bag_filename)
    except RuntimeError as err:
        assert "Invalid RealSense camera configuration, or camera not connected" in str(
            err)
