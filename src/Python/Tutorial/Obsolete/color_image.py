# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append("../..")
from py3d import *
#conda install pillow matplotlib

if __name__ == "__main__":

	print("Testing image in py3d ...")
	print("Convert an image to numpy and draw it with matplotlib.")
	x = read_image("../../TestData/image.PNG")
	print(x)
	plt.imshow(np.asarray(x))
	plt.show()

	print("Convet a numpy image to Image and show it with DrawGeomtries().")
	y = mpimg.imread("../../TestData/lena_color.jpg")
	print(y.shape)
	yy = Image(y)
	print(yy)
	draw_geometries([yy])

	print("Render a channel of the previous image.")
	z = np.array(y[:,:,1])
	print(z.shape)
	print(z.strides)
	zz = Image(z)
	print(zz)
	draw_geometries([zz])

	print("Write the previous image to file then load it with matplotlib.")
	write_image("test.jpg", zz, quality = 100)
	zzz = mpimg.imread("test.jpg")
	plt.imshow(zzz)
	plt.show()

	print("Testing basic image processing module.")
	im_raw = mpimg.imread("../../TestData/lena_color.jpg")
	im = Image(im_raw)
	im_g3 = filter_image(im, ImageFilterType.Gaussian3)
	im_g5 = filter_image(im, ImageFilterType.Gaussian5)
	im_g7 = filter_image(im, ImageFilterType.Gaussian7)
	im_gaussian = [im, im_g3, im_g5, im_g7]
	pyramid_levels = 4
	pyramid_with_gaussian_filter = True
	im_pyramid = create_image_pyramid(im, pyramid_levels,
            pyramid_with_gaussian_filter)
	im_dx = filter_image(im, ImageFilterType.Sobel3dx)
	im_dx_pyramid = filter_image_pyramid(im_pyramid, ImageFilterType.Sobel3dx)
	im_dy = filter_image(im, ImageFilterType.Sobel3dy)
	im_dy_pyramid = filter_image_pyramid(im_pyramid, ImageFilterType.Sobel3dy)
	switcher = {
		0: im_gaussian,
		1: im_pyramid,
		2: im_dx_pyramid,
		3: im_dy_pyramid,
	}
	for i in range(4):
		for j in range(pyramid_levels):
			plt.subplot(4, pyramid_levels, i*4+j+1)
			plt.imshow(switcher.get(i)[j])
	plt.show()

	print("")
