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

import open3d as o3d
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    print("Read TUM dataset")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    color_raw = o3d.io.read_image(
        dir_path + "/../../test_data/RGBD/other_formats/TUM_color.png")
    depth_raw = o3d.io.read_image(
        dir_path + "/../../test_data/RGBD/other_formats/TUM_depth.png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
        color_raw, depth_raw)
    print(rgbd_image)

    plt.subplot(1, 2, 1)
    plt.title('TUM grayscale image')
    plt.imshow(rgbd_image.color, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('TUM depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw([pcd])
