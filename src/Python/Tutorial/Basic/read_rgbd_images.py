import sys
sys.path.append("../..")

from py3d import *
import re
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
		dtype=byteorder+'u2',
		count=int(width)*int(height),
		offset=len(header)).reshape((int(height), int(width)))
	img_out = img.astype('u2')
	return img_out

if __name__ == "__main__":
    # read Redwood dataset
	color_raw_redwood = ReadImage("../../TestData/RGBD/color/00000.jpg")
	depth_raw_redwood = ReadImage("../../TestData/RGBD/depth/00000.png")
	rgbd_image_redwood = CreateRGBDImageFromColorAndDepth(
		color_raw_redwood, depth_raw_redwood);
	print(rgbd_image_redwood)
	plt.subplot(1, 2, 1)
	plt.title('Redwood grayscale image')
	plt.imshow(rgbd_image_redwood.color)
	plt.subplot(1, 2, 2)
	plt.title('Redwood depth image')
	plt.imshow(rgbd_image_redwood.depth)
	plt.show()

    # read SUN dataset
	color_raw_sun = ReadImage("../../TestData/RGBD/other_formats/SUN_color.jpg")
	depth_raw_sun = ReadImage("../../TestData/RGBD/other_formats/SUN_depth.png")
	rgbd_image_sun = CreateRGBDImageFromSUNFormat(color_raw_sun, depth_raw_sun);
	print(rgbd_image_sun)
	plt.subplot(1, 2, 1)
	plt.title('SUN grayscale image')
	plt.imshow(rgbd_image_sun.color)
	plt.subplot(1, 2, 2)
	plt.title('SUN depth image')
	plt.imshow(rgbd_image_sun.depth)
	plt.show()

    # read NYU dataset
	# Open3D does not support ppm/pgm file yet. Not using ReadImage here.
	# MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
	color_raw_nyu = mpimg.imread("../../TestData/RGBD/other_formats/NYU_color.ppm")
	depth_raw_nyu = read_nyu_pgm("../../TestData/RGBD/other_formats/NYU_depth.pgm")
	color = Image(color_raw_nyu)
	depth = Image(depth_raw_nyu)
	rgbd_image_nyu = CreateRGBDImageFromNYUFormat(color, depth);
	print(rgbd_image_nyu)
	plt.subplot(1, 2, 1)
	plt.title('NYU grayscale image')
	plt.imshow(rgbd_image_nyu.color)
	plt.subplot(1, 2, 2)
	plt.title('NYU depth image')
	plt.imshow(rgbd_image_nyu.depth)
	plt.show()

    # read TUM dataset
	color_raw_tum = ReadImage("../../TestData/RGBD/other_formats/TUM_color.png")
	depth_raw_tum = ReadImage("../../TestData/RGBD/other_formats/TUM_depth.png")
	rgbd_image_tum = CreateRGBDImageFromTUMFormat(color_raw_tum, depth_raw_tum);
	print(rgbd_image_tum)
	plt.subplot(1, 2, 1)
	plt.title('TUM grayscale image')
	plt.imshow(rgbd_image_tum.color)
	plt.subplot(1, 2, 2)
	plt.title('TUM depth image')
	plt.imshow(rgbd_image_tum.depth)
	plt.show()
