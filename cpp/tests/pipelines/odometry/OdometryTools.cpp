// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tests/pipelines/odometry/OdometryTools.h"

namespace open3d {
namespace tests {

std::shared_ptr<geometry::Image> odometry_tools::GenerateImage(
        const int& width,
        const int& height,
        const int& num_of_channels,
        const int& bytes_per_channel,
        const float& vmin,
        const float& vmax,
        const int& seed) {
    std::shared_ptr<geometry::Image> image =
            std::make_shared<geometry::Image>();

    image->Prepare(width, height, num_of_channels, bytes_per_channel);

    float* const depthData = image->PointerAs<float>();
    Rand(depthData, width * height, vmin, vmax, seed);

    return image;
}

// ----------------------------------------------------------------------------
// Shift the pixels left with a specified step.
// ----------------------------------------------------------------------------
void odometry_tools::ShiftLeft(std::shared_ptr<geometry::Image> image,
                               const int& step) {
    int width = image->width_;
    int height = image->height_;
    // int num_of_channels = image->num_of_channels_;
    // int bytes_per_channel = image->bytes_per_channel_;

    float* const float_data = image->PointerAs<float>();
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            float_data[h * width + w] =
                    float_data[h * width + (w + step) % width];
}

// ----------------------------------------------------------------------------
// Shift the pixels up with a specified step.
// ----------------------------------------------------------------------------
void odometry_tools::ShiftUp(std::shared_ptr<geometry::Image> image,
                             const int& step) {
    int width = image->width_;
    int height = image->height_;
    // int num_of_channels = image->num_of_channels_;
    // int bytes_per_channel = image->bytes_per_channel_;

    float* const float_data = image->PointerAs<float>();
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            float_data[h * width + w] =
                    float_data[((h + step) % height) * width + w];
}

// ----------------------------------------------------------------------------
// Create dummy correspondence map object.
// ----------------------------------------------------------------------------
std::shared_ptr<geometry::Image> odometry_tools::CorrespondenceMap(
        const int& width,
        const int& height,
        const int& vmin,
        const int& vmax,
        const int& seed) {
    int num_of_channels = 2;
    int bytes_per_channel = 4;

    std::shared_ptr<geometry::Image> image =
            std::make_shared<geometry::Image>();

    image->Prepare(width, height, num_of_channels, bytes_per_channel);

    int* const int_data = image->PointerAs<int>();
    size_t image_size = image->data_.size() / sizeof(int);
    Rand(int_data, image_size, vmin, vmax, seed);

    return image;
}

// ----------------------------------------------------------------------------
// Create dummy depth buffer object.
// ----------------------------------------------------------------------------
std::shared_ptr<geometry::Image> odometry_tools::DepthBuffer(const int& width,
                                                             const int& height,
                                                             const float& vmin,
                                                             const float& vmax,
                                                             const int& seed) {
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    std::shared_ptr<geometry::Image> image =
            std::make_shared<geometry::Image>();

    image->Prepare(width, height, num_of_channels, bytes_per_channel);

    float* const float_data = image->PointerAs<float>();
    size_t image_size = image->data_.size() / sizeof(float);
    Rand(float_data, image_size, vmin, vmax, seed);

    return image;
}

}  // namespace tests
}  // namespace open3d
