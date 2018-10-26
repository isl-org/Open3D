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

#include "OdometryTools.h"

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
shared_ptr<Image> odometry_tools::GenerateImage(const int& width,
                                                const int& height,
                                                const int& num_of_channels,
                                                const int& bytes_per_channel,
                                                const float& vmin,
                                                const float& vmax,
                                                const int& seed)
{
    Image image;

    image.PrepareImage(width,
                       height,
                       num_of_channels,
                       bytes_per_channel);

    float* const depthData = reinterpret_cast<float*>(&image.data_[0]);
    Rand(depthData, width * height, vmin, vmax, seed);

    return make_shared<Image>(image);
}

// ----------------------------------------------------------------------------
// Shift the pixels left with a specified step.
// ----------------------------------------------------------------------------
void odometry_tools::ShiftLeft(shared_ptr<Image> image, const int& step)
{
    int width = image->width_;
    int height = image->height_;
    int num_of_channels = image->num_of_channels_;
    int bytes_per_channel = image->bytes_per_channel_;

    float* const floatData = reinterpret_cast<float*>(&image->data_[0]);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            floatData[h * width + w] = floatData[h * width + (w + step) % width];
}

// ----------------------------------------------------------------------------
// Shift the pixels up with a specified step.
// ----------------------------------------------------------------------------
void odometry_tools::ShiftUP(shared_ptr<Image> image, const int& step)
{
    int width = image->width_;
    int height = image->height_;
    int num_of_channels = image->num_of_channels_;
    int bytes_per_channel = image->bytes_per_channel_;

    float* const floatData = reinterpret_cast<float*>(&image->data_[0]);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            floatData[h * width + w] = floatData[((h + step) % height) * width + w];
}
