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

#include "open3d/tgeometry/Image.h"

#include "core/CoreTest.h"
#include "open3d/core/TensorList.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class ImagePermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Image,
                         ImagePermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class ImagePermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Image,
        ImagePermuteDevicePairs,
        testing::ValuesIn(ImagePermuteDevicePairs::TestCases()));

TEST_P(ImagePermuteDevices, ConstructorNoArg) {
    tgeometry::Image im;
    EXPECT_EQ(im.GetRows(), 0);
    EXPECT_EQ(im.GetCols(), 0);
    EXPECT_EQ(im.GetChannels(), 1);
    EXPECT_EQ(im.GetDtype(), core::Dtype::Float32);
    EXPECT_EQ(im.GetDevice(), core::Device("CPU:0"));
}

TEST_P(ImagePermuteDevices, Constructor) {
    core::Device device = GetParam();

    // Normal case.
    int64_t rows = 480;
    int64_t cols = 640;
    int64_t channels = 3;
    core::Dtype dtype = core::Dtype::UInt8;
    tgeometry::Image im(rows, cols, channels, dtype, device);
    EXPECT_EQ(im.GetRows(), rows);
    EXPECT_EQ(im.GetCols(), cols);
    EXPECT_EQ(im.GetChannels(), channels);
    EXPECT_EQ(im.GetDtype(), dtype);
    EXPECT_EQ(im.GetDevice(), device);

    // Unsupported shape or channel.
    EXPECT_ANY_THROW(tgeometry::Image(-1, cols, channels, dtype, device));
    EXPECT_ANY_THROW(tgeometry::Image(rows, -1, channels, dtype, device));
    EXPECT_ANY_THROW(tgeometry::Image(rows, cols, 0, dtype, device));
    EXPECT_ANY_THROW(tgeometry::Image(rows, cols, -1, dtype, device));

    // Unsupported dtypes.
    EXPECT_NO_THROW(
            tgeometry::Image(rows, cols, channels, core::Dtype::UInt8, device));
    EXPECT_NO_THROW(tgeometry::Image(rows, cols, channels, core::Dtype::UInt16,
                                     device));
    EXPECT_NO_THROW(tgeometry::Image(rows, cols, channels, core::Dtype::Float32,
                                     device));
    EXPECT_NO_THROW(tgeometry::Image(rows, cols, channels, core::Dtype::Float64,
                                     device));
    EXPECT_ANY_THROW(
            tgeometry::Image(rows, cols, channels, core::Dtype::Int64, device));
}

}  // namespace tests
}  // namespace open3d
