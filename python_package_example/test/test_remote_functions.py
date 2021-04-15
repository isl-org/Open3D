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

import os
import open3d as o3d
import numpy as np
import pytest

# skip all tests if the RPC interface was not built
pytestmark = pytest.mark.skipif(not o3d._build_config['BUILD_RPC_INTERFACE'],
                                reason='rpc interface not built.')
if os.name == 'nt':
    address = 'tcp://127.0.0.1:51455'
else:
    address = 'ipc:///tmp/open3d_ipc'


def test_external_visualizer():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

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
