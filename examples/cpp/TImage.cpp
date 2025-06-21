// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d::utility;

    open3d::PrintOpen3DVersion();
    // clang-format off
    LogInfo("Usage:");
    LogInfo("    > TImageOpenCV IMAGE_FILENAME ACTION DATA");
    LogInfo("    where ACTION can be:");
    LogInfo("    1) Colorize depth image with Open3D, display with OpenCV, and save to disk with Open3D.");
    LogInfo("    3) Draw a bounding box on the image and display it with OpenCV.");
    LogInfo("    5) Save the two processed images with Open3D.");
    // clang-format on
    LogInfo("");
}

/// Display an image with OpenCV in a winodow with the given title and wait for
/// the user to close it.
void imshowWait(const std::string &title, const cv::Mat &im) {
    cv::imshow(title, im);
    cv::waitKey(0);
    cv::destroyWindow(title);
}

int main(int argc, char *argv[]) {
    fmt::print("Reading image {}\n", argv[1]);
    // Read an image without an conversion with OpenCV
    cv::Mat im_cv = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (im_cv.empty()) {
        open3d::utility::LogError("Could not open image!");
    }
    fmt::print("rows={}, cols={}, channels={}, depth={}\n", im_cv.rows,
               im_cv.cols, im_cv.channels(), im_cv.depth());
    // Convert OpenCV depth to Open3D Dtype
    open3d::core::Dtype dt;
    switch (im_cv.depth()) {
        case CV_8U:
            dt = open3d::core::Dtype::UInt8;
            break;
        case CV_16U:
            dt = open3d::core::Dtype::UInt16;
            break;
        case CV_32F:
            dt = open3d::core::Dtype::Float32;
            break;
        case CV_64F:
            dt = open3d::core::Dtype::Float64;
            break;
        default:
            open3d::utility::LogError("See docs for other element types.");
    }
    // Create a blob for externally managed memory with no-op deleter
    auto pblob = std::make_shared<open3d::core::Blob>(
            open3d::core::Device(), im_cv.data, [](void *) {});
    // Create tensor
    open3d::core::Tensor data_o3d(
            /*shape=*/{im_cv.rows, im_cv.cols, im_cv.channels()},
            /*stride in elements (not bytes)*/
            {int64_t(im_cv.step[0] / im_cv.elemSize1()),
             int64_t(im_cv.step[1] / im_cv.elemSize1()), 1},
            im_cv.data, dt, pblob);
    open3d::t::geometry::Image im_o3d(data_o3d);
    // If you need the Eigen based legacy Image:
    // open3d::geometry::Image im_o3d_e = im_o3d.ToLegacy();
    fmt::print("rows={}, cols={}, channels={}, dtype={}\n", im_o3d.GetRows(),
               im_o3d.GetCols(), im_o3d.GetChannels(),
               im_o3d.GetDtype().ToString());
    open3d::t::io::WriteImage("output_image.png", im_o3d);
}
/* using namespace open3d::utility; */
/* using namespace open3d::t::geometry; */
/* using namespace open3d::t::io; */

/* SetVerbosityLevel(VerbosityLevel::Debug); */

/* if (argc != 3 || ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) { */
/*     PrintHelp(); */
/*     return 1; */
/* } */

/* const std::string filename_rgb(argv[1]); */
/* const std::string filename_depth(argv[2]); */

/* Image color_image; */
/* if (!ReadImage(filename_rgb, color_image)) { */
/*     LogWarning("Failed to read {}", filename_rgb); */
/*     return 1; */
/* } */
/* cv::Mat color_mat(color_image.GetRows(), color_image.GetCols(), CV_8UC3, */
/*                   color_image.GetDataPtr()); */
/* imshowWait(filename_rgb + " : " color_image.ToString(), color_mat); */

/* auto bf_color_image = color_image.FilterBilateral(9, 20.f, 31.f); */
/* cv::Mat bf_color_mat(bf_color_image.GetRows(), bf_color_image.GetCols(), */
/*                      CV_8UC3, bf_color_image.GetDataPtr()); */
/* cv::rectangle(bf_color_mat, cv::Point(620, 356), cv::Point(670, 475), */
/*               cv::Scalar(0, 255, 0), 4); */
/* imshowWait("RGB image with bilateral filter and bounding box", */
/*            bf_color_mat); */

/* WriteImage("bilateral_filtered_color_image.jpg", bf_color_image); */

/* LogDebug( */
/*         "Bilateral Filtered Image written to " */
/*         "bilateral_filtered_color_image.jpg"); */

/* Image depth_image_16bit; */
/* if (ReadImage(filename_depth, depth_image_16bit)) { */
/*     LogDebug("Depth image {} : {}", filename_depth, */
/*              depth_image_16bit.ToString()); */
/*     uint16_t m = depth_image_16bit.AsTensor() */
/*                          .Min({0, 1, 2}) */
/*                          .template Item<uint16_t>(), */
/*              M = depth_image_16bit.AsTensor() */
/*                          .Max({0, 1, 2}) */
/*                          .template Item<uint16_t>(); */
/*     float scale = M > m ? (M - m) / 255. : 1.f; */
/*     auto colorized_image = depth_image_16bit.ColorizeDepth(scale, m, M); */
/*     cv::Mat depth_mat(colorized_image.GetRows(), colorized_image.GetCols(),
 */
/*                       CV_8UC3, colorized_image.GetDataPtr()); */

/*     cv::imshow("Colorized depth image", depth_mat); */
/*     cv::waitKey(0); */
/*     cv::destroyAllWindows(); */
/*     WriteImage("colorized_depth_image.jpg", colorized_image); */
/*     LogDebug("Colorized depth image written to colorized_depth_image.jpg");
 */
/* } else { */
/*     LogWarning("Failed to read {}", filename_depth); */
/*     return 1; */
/* } */

/* return 0; */
/* } */
