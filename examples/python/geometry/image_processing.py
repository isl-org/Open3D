# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import open3d as o3d
#conda install pillow matplotlib

if __name__ == "__main__":

    print("Testing image in open3d ...")
    print("Convert an image to numpy")
    sample_image = o3d.data.JuneauImage()
    x = o3d.io.read_image(sample_image.path)
    print(np.asarray(x))
    print(
        "Convert a numpy image to o3d.geometry.Image and show it with DrawGeomtries()."
    )
    y = mpimg.imread(sample_image.path)
    print(y.shape)
    yy = o3d.geometry.Image(y)
    print(yy)
    o3d.visualization.draw_geometries([yy])

    print("Render a channel of the previous image.")
    z = np.array(y[:, :, 1])
    print(z.shape)
    print(z.strides)
    zz = o3d.geometry.Image(z)
    print(zz)
    o3d.visualization.draw_geometries([zz])

    print("Write the previous image to file.")
    o3d.io.write_image("test.jpg", zz, quality=100)

    print("Testing basic image processing module.")
    sample_image = o3d.data.JuneauImage()
    im_raw = mpimg.imread(sample_image.path)
    im = o3d.geometry.Image(im_raw)
    im_g3 = im.filter(o3d.geometry.ImageFilterType.Gaussian3)
    im_g5 = im.filter(o3d.geometry.ImageFilterType.Gaussian5)
    im_g7 = im.filter(o3d.geometry.ImageFilterType.Gaussian7)
    im_gaussian = [im, im_g3, im_g5, im_g7]
    pyramid_levels = 4
    pyramid_with_gaussian_filter = True
    im_pyramid = im.create_pyramid(pyramid_levels, pyramid_with_gaussian_filter)
    im_dx = im.filter(o3d.geometry.ImageFilterType.Sobel3dx)
    im_dx_pyramid = o3d.geometry.Image.filter_pyramid(
        im_pyramid, o3d.geometry.ImageFilterType.Sobel3dx)
    im_dy = im.filter(o3d.geometry.ImageFilterType.Sobel3dy)
    im_dy_pyramid = o3d.geometry.Image.filter_pyramid(
        im_pyramid, o3d.geometry.ImageFilterType.Sobel3dy)
    switcher = {
        0: im_gaussian,
        1: im_pyramid,
        2: im_dx_pyramid,
        3: im_dy_pyramid,
    }
    for i in range(4):
        for j in range(pyramid_levels):
            plt.subplot(4, pyramid_levels, i * 4 + j + 1)
            plt.imshow(switcher.get(i)[j])
    plt.show()
