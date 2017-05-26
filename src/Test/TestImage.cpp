// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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

	SetVerbosityLevel(three::VERBOSE_ALWAYS);
	
	if (argc != 3) {
		PrintInfo("Usage:\n");
		PrintInfo("    > TestImage [image filename] [depth filename]\n");
		PrintInfo("    The program will :\n");
		PrintInfo("    1) Read 8bit color or 16bit depth image\n");
		PrintInfo("    2) Convert image to single channel float image\n");
		PrintInfo("    3) Making image pyramid that includes Gaussian blur and downsampling\n");
		PrintInfo("    4) Will save all the layers in the image pyramid\n");
		return 0;
	}

	const std::string filename_color(argv[1]);
	const std::string filename_depth(argv[2]);

	Image color; // check the name
	if (ReadImage(filename_color, color)) {
		PrintDebug("Color image size : %d x %d\n", color.width_, color.height_);
		auto color_f = CreateFloatImageFromImage(color);
		auto color_fo = TypecastImage<uint8_t>(*color_f);
		WriteImage("test.png", *color_fo);

		auto color_fb3 = FilterImage(*color_f, FILTER_GAUSSIAN_3);
		auto color_fb3o = TypecastImage<uint8_t>(*color_fb3);
		WriteImage("test_blur3.png", *color_fb3o);

		auto color_fb5 = FilterImage(*color_f, FILTER_GAUSSIAN_5);
		auto color_fb5o = TypecastImage<uint8_t>(*color_fb5);
		WriteImage("test_blur5.png", *color_fb5o);

		auto color_fb7 = FilterImage(*color_f, FILTER_GAUSSIAN_7);
		auto color_fb7o = TypecastImage<uint8_t>(*color_fb7);
		WriteImage("test_blur7.png", *color_fb7o);

		auto color_pyramid = CreateImagePyramid(*color_f, 4);
		for (int i = 0; i < 4; i++) {
			auto layer = color_pyramid[i];
			auto layer8 = TypecastImage<uint8_t>(*layer);
			std::string outputname = 
				"test_" + std::to_string(i) + ".png";
			WriteImage(outputname, *layer8);
		}
	} else {
		PrintError("Failed to read %s\n\n", filename_color);
	}

	//Image depth; // check the name
	//if (ReadImage(filename_depth, depth)) {
	//	PrintDebug("Depth image size : %d x %d\n", depth.width_, depth.height_);
	//	Image depth_f;
	//	ConvertDepthToFloatImage(depth, depth_f);	
	//	auto depth_pyramid = CreateImagePyramid(depth_f, 4);
	//}
	//else {
	//	PrintError("Failed to read %s\n\n", filename_color);
	//}

	return 0;
}
