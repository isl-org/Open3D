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

#include "Open3D/Open3D.h"

int main(int argc, char **argv) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3) {
        PrintOpen3DVersion();
        // clang-format off
        utility::LogInfo("Usage:");
        utility::LogInfo("    > Image [image filename] [depth filename]");
        utility::LogInfo("    The program will :");
        utility::LogInfo("    1) Read 8bit RGB and 16bit depth image");
        utility::LogInfo("    2) Convert RGB image to single channel float image");
        utility::LogInfo("    3) 3x3, 5x5, 7x7 Gaussian filters are applied");
        utility::LogInfo("    4) 3x3 Sobel filter for x-and-y-directions are applied");
        utility::LogInfo("    5) Make image pyramid that includes Gaussian blur and downsampling");
        utility::LogInfo("    6) Will save all the layers in the image pyramid");
        // clang-format on
        return 1;
    }

    const std::string filename_rgb(argv[1]);
    const std::string filename_depth(argv[2]);

    geometry::Image color_image_8bit;
    if (io::ReadImage(filename_rgb, color_image_8bit)) {
        utility::LogDebug("RGB image size : {:d} x {:d}",
                          color_image_8bit.width_, color_image_8bit.height_);
        auto gray_image = color_image_8bit.CreateFloatImage();
        io::WriteImage("gray.png",
                       *gray_image->CreateImageFromFloatImage<uint8_t>());

        utility::LogDebug("Gaussian Filtering");
        auto gray_image_b3 =
                gray_image->Filter(geometry::Image::FilterType::Gaussian3);
        io::WriteImage("gray_blur3.png",
                       *gray_image_b3->CreateImageFromFloatImage<uint8_t>());
        auto gray_image_b5 =
                gray_image->Filter(geometry::Image::FilterType::Gaussian5);
        io::WriteImage("gray_blur5.png",
                       *gray_image_b5->CreateImageFromFloatImage<uint8_t>());
        auto gray_image_b7 =
                gray_image->Filter(geometry::Image::FilterType::Gaussian7);
        io::WriteImage("gray_blur7.png",
                       *gray_image_b7->CreateImageFromFloatImage<uint8_t>());

        utility::LogDebug("Sobel Filtering");
        auto gray_image_dx =
                gray_image->Filter(geometry::Image::FilterType::Sobel3Dx);
        // make [-1,1] to [0,1].
        gray_image_dx->LinearTransform(0.5, 0.5);
        gray_image_dx->ClipIntensity();
        io::WriteImage("gray_sobel_dx.png",
                       *gray_image_dx->CreateImageFromFloatImage<uint8_t>());
        auto gray_image_dy =
                gray_image->Filter(geometry::Image::FilterType::Sobel3Dy);
        gray_image_dy->LinearTransform(0.5, 0.5);
        gray_image_dy->ClipIntensity();
        io::WriteImage("gray_sobel_dy.png",
                       *gray_image_dy->CreateImageFromFloatImage<uint8_t>());

        utility::LogDebug("Build Pyramid");
        auto pyramid = gray_image->CreatePyramid(4);
        for (int i = 0; i < 4; i++) {
            auto level = pyramid[i];
            auto level_8bit = level->CreateImageFromFloatImage<uint8_t>();
            std::string outputname =
                    "gray_pyramid_level" + std::to_string(i) + ".png";
            io::WriteImage(outputname, *level_8bit);
        }
    } else {
        utility::LogWarning("Failed to read {}", filename_rgb);
        return 1;
    }

    geometry::Image depth_image_16bit;
    if (io::ReadImage(filename_depth, depth_image_16bit)) {
        utility::LogDebug("Depth image size : {:d} x {:d}",
                          depth_image_16bit.width_, depth_image_16bit.height_);
        auto depth_image = depth_image_16bit.CreateFloatImage();
        io::WriteImage("depth.png",
                       *depth_image->CreateImageFromFloatImage<uint16_t>());

        utility::LogDebug("Gaussian Filtering");
        auto depth_image_b3 =
                depth_image->Filter(geometry::Image::FilterType::Gaussian3);
        io::WriteImage("depth_blur3.png",
                       *depth_image_b3->CreateImageFromFloatImage<uint16_t>());
        auto depth_image_b5 =
                depth_image->Filter(geometry::Image::FilterType::Gaussian5);
        io::WriteImage("depth_blur5.png",
                       *depth_image_b5->CreateImageFromFloatImage<uint16_t>());
        auto depth_image_b7 =
                depth_image->Filter(geometry::Image::FilterType::Gaussian7);
        io::WriteImage("depth_blur7.png",
                       *depth_image_b7->CreateImageFromFloatImage<uint16_t>());

        utility::LogDebug("Sobel Filtering");
        auto depth_image_dx =
                depth_image->Filter(geometry::Image::FilterType::Sobel3Dx);
        // make [-65536,65536] to [0,13107.2]. // todo: need to test this
        depth_image_dx->LinearTransform(0.1, 6553.6);
        depth_image_dx->ClipIntensity(0.0, 13107.2);
        io::WriteImage("depth_sobel_dx.png",
                       *depth_image_dx->CreateImageFromFloatImage<uint16_t>());
        auto depth_image_dy =
                depth_image->Filter(geometry::Image::FilterType::Sobel3Dy);
        depth_image_dy->LinearTransform(0.1, 6553.6);
        depth_image_dx->ClipIntensity(0.0, 13107.2);
        io::WriteImage("depth_sobel_dy.png",
                       *depth_image_dy->CreateImageFromFloatImage<uint16_t>());

        utility::LogDebug("Build Pyramid");
        auto pyramid = depth_image->CreatePyramid(4);
        for (int i = 0; i < 4; i++) {
            auto level = pyramid[i];
            auto level_16bit = level->CreateImageFromFloatImage<uint16_t>();
            std::string outputname =
                    "depth_pyramid_level" + std::to_string(i) + ".png";
            io::WriteImage(outputname, *level_16bit);
        }
    } else {
        utility::LogWarning("Failed to read {}", filename_depth);
        return 1;
    }

    return 0;
}
