# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import socket
import open3d as o3d
import numpy as np
import pytest


def _get_test_address():
    if os.name != 'nt':
        return 'ipc:///tmp/open3d_ipc'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
        test_socket.bind(('127.0.0.1', 0))
        return f'tcp://127.0.0.1:{test_socket.getsockname()[1]}'


def test_external_visualizer():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    address = _get_test_address()

    # create dummy receiver which will receive all data
    receiver = o3d.io.rpc._DummyReceiver(address=address)
    receiver.start()

    # create ev with the same address
    ev = o3d.visualization.ExternalVisualizer(address=address)

    # create some objects
    mesh = o3d.geometry.TriangleMesh.create_torus()
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.random.rand(100, 3)))
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = np.eye(4)

    # send single objects
    assert ev.set(pcd, path='bla/pcd', time=42)
    assert ev.set(mesh, path='bla/mesh', time=42)
    assert ev.set(camera, path='bla/camera', time=42)

    # send multiple objects
    assert ev.set(obj=[pcd, mesh, camera])

    # send multiple objects with args
    assert ev.set(obj=[(pcd, 'pcd', 1), (mesh, 'mesh', 2), (camera, 'camera',
                                                            3)])

    # test other commands
    ev.set_time(10)
    ev.set_active_camera('camera')

    receiver.stop()
