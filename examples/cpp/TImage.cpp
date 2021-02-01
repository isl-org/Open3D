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

// Only for testing purpose. To be deleted before merging with master.

#include <cstdio>

#include "open3d/Open3D.h"

using namespace open3d;

int main(int argc, char **argv) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    const core::Device device = core::Device(argv[1]);
    const std::string filename_rgb(argv[2]);
    // const std::string filename_depth(argv[2]);

    // As device transfer is currently not supported in geometry::Image
    // and IO does not support CUDA, therefore copy to legacy image
    // for checking output.
    geometry::Image output_legacy;

    t::geometry::Image color_image(0, 0, 3, core::Dtype::UInt16, device);
    if (t::io::ReadImage(filename_rgb, color_image)) {
        utility::LogDebug("RGB image size : {:d} x {:d}", color_image.GetRows(),
                          color_image.GetCols());

        // Saving original image
        utility::LogDebug("Saving original input RGB image");
        output_legacy = color_image.ToLegacyImage();
        io::WriteImageToPNG("output01.png", output_legacy);
        utility::LogDebug("    Saved");

        // Dilate operation
        utility::LogDebug("Dilate Image");
        int half_kernel_size = 1;
        t::geometry::Image dilate_image = color_image.Dilate(half_kernel_size);
        output_legacy = dilate_image.ToLegacyImage();
        utility::LogDebug("Saving Dilated output image");
        io::WriteImageToPNG("output02.png", output_legacy);
        utility::LogDebug("    Saved");

        // Gaussian operation
        utility::LogDebug("Gaussian Filter");
        t::geometry::Image gauss_image = color_image.Filter(
                t::geometry::Image::FilterType::Gaussian15x15);
        output_legacy = gauss_image.ToLegacyImage();
        utility::LogDebug("Saving Gaussian output filter");
        io::WriteImageToPNG("output03.png", output_legacy);
        utility::LogDebug("    Saved");

        // Sobel operation
        utility::LogDebug("Sobel Horizontal Filter");
        t::geometry::Image sobelx_image = color_image.Filter(
                t::geometry::Image::FilterType::SobelHorizontal);
        output_legacy = sobelx_image.ToLegacyImage();
        utility::LogDebug("Saving Sobel-horizontal filter output.");
        io::WriteImageToPNG("output04.png", output_legacy);
        utility::LogDebug("    Saved");

        utility::LogDebug("Sobel Vertical Filter");
        t::geometry::Image sobely_image = color_image.Filter(
                t::geometry::Image::FilterType::SobelVertical);
        output_legacy = sobely_image.ToLegacyImage();
        utility::LogDebug("Saving Sobel-vertical filter output.");
        io::WriteImageToPNG("output05.png", output_legacy);
        utility::LogDebug("    Saved");

        // Bilateral operation
        // utility::LogDebug("Bilateral Filter");
        // t::geometry::Image bilateral_image =
        // color_image.Filter(t::geometry::Image::FilterType::BilateralGauss);
        // output_legacy = bilateral_image.ToLegacyImage();
        // utility::LogDebug("Saving Sobel-vertical filter output.");
        // io::WriteImageToPNG("output06.png", output_legacy);
        // utility::LogDebug("    Saved");

        utility::LogDebug("All operations performed successfully...");
    }
    return 0;
}
