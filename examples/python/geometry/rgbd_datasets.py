# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import matplotlib.image as mpimg
import re


def visualize_rgbd(rgbd_image):
    print(rgbd_image)

    o3d.visualization.draw_geometries([rgbd_image])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


def nyu_dataset():
    print("Read NYU dataset")
    # Open3D does not support ppm/pgm file yet. Not using o3d.io.read_image here.
    # MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
    nyu_data = o3d.data.SampleNYURGBDImage()
    color_raw = mpimg.imread(nyu_data.color_path)
    depth_raw = read_nyu_pgm(nyu_data.depth_path)
    color = o3d.geometry.Image(color_raw)
    depth = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(
        color, depth, convert_rgb_to_intensity=False)

    print("Displaying NYU color and depth images and pointcloud ...")
    visualize_rgbd(rgbd_image)


def redwood_dataset():
    print("Read Redwood dataset")
    redwood_data = o3d.data.SampleRedwoodRGBDImages()
    color_raw = o3d.io.read_image(redwood_data.color_paths[0])
    depth_raw = o3d.io.read_image(redwood_data.depth_paths[0])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

    print("Displaying Redwood color and depth images and pointcloud ...")
    visualize_rgbd(rgbd_image)


def sun_dataset():
    print("Read SUN dataset")
    sun_data = o3d.data.SampleSUNRGBDImage()
    color_raw = o3d.io.read_image(sun_data.color_path)
    depth_raw = o3d.io.read_image(sun_data.depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

    print("Displaying SUN color and depth images and pointcloud ...")
    visualize_rgbd(rgbd_image)


def tum_dataset():
    print("Read TUM dataset")
    tum_data = o3d.data.SampleTUMRGBDImage()
    color_raw = o3d.io.read_image(tum_data.color_path)
    depth_raw = o3d.io.read_image(tum_data.depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

    print("Displaying TUM color and depth images and pointcloud ...")
    visualize_rgbd(rgbd_image)


if __name__ == "__main__":
    nyu_dataset()
    redwood_dataset()
    sun_dataset()
    tum_dataset()
