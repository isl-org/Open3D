// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <cstdio>

#include <Core/Core.h>
#include <IO/IO.h>

int main(int argc, char **argv)
{
	using namespace three;

	SetVerbosityLevel(VerbosityLevel::VerboseAlways);

	if (argc != 3) {
		PrintInfo("Usage:\n");
		PrintInfo("    > TestImage [image filename] [depth filename]\n");
		PrintInfo("    The program will :\n");
		PrintInfo("    1) Read 8bit RGB and 16bit depth image\n");
		PrintInfo("    2) Convert RGB image to single channel float image\n");
		PrintInfo("    3) 3x3, 5x5, 7x7 Gaussian filters are applied\n");
		PrintInfo("    4) 3x3 Sobel filter for x-and-y-directions are applied\n");
		PrintInfo("    5) Make image pyramid that includes Gaussian blur and downsampling\n");
		PrintInfo("    6) Will save all the layers in the image pyramid\n");
		return 0;
	}

	const std::string filename_rgb(argv[1]);
	const std::string filename_depth(argv[2]);

	Image color_image_8bit;
	if (ReadImage(filename_rgb, color_image_8bit)) {

		PrintDebug("RGB image size : %d x %d\n",
				color_image_8bit.width_, color_image_8bit.height_);
		auto gray_image = CreateFloatImageFromImage(color_image_8bit);
		WriteImage("gray.png",
				*CreateImageFromFloatImage<uint8_t>(*gray_image));

		PrintDebug("Gaussian Filtering\n");
		auto gray_image_b3 = FilterImage(*gray_image,
				Image::FilterType::Gaussian3);
		WriteImage("gray_blur3.png",
				*CreateImageFromFloatImage<uint8_t>(*gray_image_b3));
		auto gray_image_b5 = FilterImage(*gray_image,
				Image::FilterType::Gaussian5);
		WriteImage("gray_blur5.png",
				*CreateImageFromFloatImage<uint8_t>(*gray_image_b5));
		auto gray_image_b7 = FilterImage(*gray_image,
				Image::FilterType::Gaussian7);
		WriteImage("gray_blur7.png",
				*CreateImageFromFloatImage<uint8_t>(*gray_image_b7));

		PrintDebug("Sobel Filtering\n");
		auto gray_image_dx = FilterImage(*gray_image,
				Image::FilterType::Sobel3Dx);
		// make [-1,1] to [0,1].
		LinearTransformImage(*gray_image_dx, 0.5, 0.5);
		ClipIntensityImage(*gray_image_dx);
		WriteImage("gray_sobel_dx.png",
				*CreateImageFromFloatImage<uint8_t>(*gray_image_dx));
		auto gray_image_dy = FilterImage(*gray_image,
				Image::FilterType::Sobel3Dy);
		LinearTransformImage(*gray_image_dy, 0.5, 0.5);
		ClipIntensityImage(*gray_image_dy);
		WriteImage("gray_sobel_dy.png",
				*CreateImageFromFloatImage<uint8_t>(*gray_image_dy));

		PrintDebug("Build Pyramid\n");
		auto pyramid = CreateImagePyramid(*gray_image, 4);
		for (int i = 0; i < 4; i++) {
			auto level = pyramid[i];
			auto level_8bit = CreateImageFromFloatImage<uint8_t>(*level);
			std::string outputname =
				"gray_pyramid_level" + std::to_string(i) + ".png";
			WriteImage(outputname, *level_8bit);
		}
	} else {
		PrintError("Failed to read %s\n\n", filename_rgb.c_str());
	}

	Image depth_image_16bit;
	if (ReadImage(filename_depth, depth_image_16bit)) {

		PrintDebug("Depth image size : %d x %d\n",
			depth_image_16bit.width_, depth_image_16bit.height_);
		auto depth_image = CreateFloatImageFromImage(depth_image_16bit);
		WriteImage("depth.png",
				*CreateImageFromFloatImage<uint16_t>(*depth_image));

		PrintDebug("Gaussian Filtering\n");
		auto depth_image_b3 = FilterImage(*depth_image,
				Image::FilterType::Gaussian3);
		WriteImage("depth_blur3.png",
				*CreateImageFromFloatImage<uint16_t>(*depth_image_b3));
		auto depth_image_b5 = FilterImage(*depth_image,
				Image::FilterType::Gaussian5);
		WriteImage("depth_blur5.png",
				*CreateImageFromFloatImage<uint16_t>(*depth_image_b5));
		auto depth_image_b7 = FilterImage(*depth_image,
				Image::FilterType::Gaussian7);
		WriteImage("depth_blur7.png",
				*CreateImageFromFloatImage<uint16_t>(*depth_image_b7));

		PrintDebug("Sobel Filtering\n");
		auto depth_image_dx = FilterImage(*depth_image,
				Image::FilterType::Sobel3Dx);
		// make [-65536,65536] to [0,13107.2]. // todo: need to test this
		LinearTransformImage(*depth_image_dx, 0.1, 6553.6);
		ClipIntensityImage(*depth_image_dx, 0.0, 13107.2);
		WriteImage("depth_sobel_dx.png",
				*CreateImageFromFloatImage<uint16_t>(*depth_image_dx));
		auto depth_image_dy = FilterImage(*depth_image,
				Image::FilterType::Sobel3Dy);
		LinearTransformImage(*depth_image_dy, 0.1, 6553.6);
		ClipIntensityImage(*depth_image_dx, 0.0, 13107.2);
		WriteImage("depth_sobel_dy.png",
				*CreateImageFromFloatImage<uint16_t>(*depth_image_dy));

		PrintDebug("Build Pyramid\n");
		auto pyramid = CreateImagePyramid(*depth_image, 4);
		for (int i = 0; i < 4; i++) {
			auto level = pyramid[i];
			auto level_16bit = CreateImageFromFloatImage<uint16_t>(*level);
			std::string outputname =
					"depth_pyramid_level" + std::to_string(i) + ".png";
			WriteImage(outputname, *level_16bit);
		}
	}
	else {
		PrintError("Failed to read %s\n\n", filename_depth.c_str());
	}

	return 0;
}
