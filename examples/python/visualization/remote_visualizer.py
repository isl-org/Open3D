# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""This example shows Open3D's remote visualization feature using RPC
communication. To run this example, start the client first by running

python remote_visualizer.py client

and then run the server by running

python remote_visualizer.py server

Port 51454 is used by default for communication. For remote visualization (client
and server running on different machines), use ssh to forward the remote server
port to your local computer:

    ssh -N -R 51454:localhost:51454 user@remote_host

See documentation for more details (e.g. to use a different port).
"""
import sys
import numpy as np
import open3d as o3d
import open3d.visualization as vis


def make_point_cloud(npts, center, radius, colorize):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colorize:
        colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def server_time_animation():
    orig = make_point_cloud(200, (0, 0, 0), 1.0, True)
    clouds = [{"name": "t=0", "geometry": orig, "time": 0}]
    drift_dir = (1.0, 0.0, 0.0)
    expand = 1.0
    n = 20
    ev = o3d.visualization.ExternalVisualizer()
    for i in range(1, n):
        amount = float(i) / float(n - 1)
        cloud = o3d.geometry.PointCloud()
        pts = np.asarray(orig.points)
        pts = pts * (1.0 + amount * expand) + [amount * v for v in drift_dir]
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = orig.colors
        ev.set(obj=cloud, time=i, path=f"points at t={i}")
        print('.', end='', flush=True)
    print()


def client_time_animation():
    o3d.visualization.draw(title="Open3D - Remote Visualizer Client",
                           show_ui=True,
                           rpc_interface=True)


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ('client', 'server'), (
        "Usage: python remote_visualizer.py [client|server]")
    if sys.argv[1] == "client":
        client_time_animation()
    elif sys.argv[1] == "server":
        server_time_animation()
