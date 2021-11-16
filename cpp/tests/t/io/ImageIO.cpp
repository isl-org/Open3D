// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/io/ImageIO.h"

#include <gtest/gtest.h>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

t::geometry::Image CreateTestImage() {
    core::Tensor t(core::SizeVector{150, 100, 3}, core::UInt8);
    t.Slice(2, 0, 1).Fill(250);
    t.Slice(2, 1, 2).Fill(150);
    t.Slice(2, 2, 3).Fill(200);

    return t::geometry::Image(t);
}

void WriteTestImage(t::geometry::Image image) {
    t::io::WriteImage(utility::GetDataPathCommon("test_imageio.png"), image);
    t::io::WriteImage(utility::GetDataPathCommon("test_imageio.jpg"), image);
}

int RemoveTestImage(std::string filename) {
    return std::remove(filename.c_str());
}

}  // namespace

// Write test image.
TEST(ImageIO, WriteImage) {
    t::geometry::Image test_img = CreateTestImage();
    EXPECT_TRUE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio.png"), test_img));
    EXPECT_TRUE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio.jpg"), test_img));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

TEST(ImageIO, CreateImageFromFile) {
    WriteTestImage(CreateTestImage());
    std::shared_ptr<t::geometry::Image> img_png = t::io::CreateImageFromFile(
            utility::GetDataPathCommon("test_imageio.png"));
    std::shared_ptr<t::geometry::Image> img_jpg = t::io::CreateImageFromFile(
            utility::GetDataPathCommon("test_imageio.jpg"));

    EXPECT_EQ(img_png->GetRows(), 150);
    EXPECT_EQ(img_png->GetCols(), 100);
    EXPECT_EQ(img_png->GetChannels(), 3);
    EXPECT_EQ(img_png->GetDtype(), core::UInt8);
    EXPECT_EQ(img_png->GetDevice(), img_jpg->GetDevice());

    t::geometry::Image test_img = CreateTestImage();

    EXPECT_EQ(img_jpg->GetRows(), 150);
    EXPECT_EQ(img_jpg->GetCols(), 100);
    EXPECT_EQ(img_jpg->GetChannels(), 3);
    EXPECT_EQ(img_jpg->GetDtype(), core::UInt8);
    EXPECT_EQ(img_jpg->GetDevice(), test_img.GetDevice());

    EXPECT_FALSE(img_jpg->AsTensor().IsSame(test_img.AsTensor()));
    EXPECT_TRUE(img_jpg->AsTensor().AllClose(test_img.AsTensor()));
    EXPECT_TRUE(img_png->AsTensor().AllClose(test_img.AsTensor()));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

TEST(ImageIO, ReadImage) {
    WriteTestImage(CreateTestImage());
    t::geometry::Image img;
    EXPECT_TRUE(t::io::ReadImage(utility::GetDataPathCommon("test_imageio.png"),
                                 img));
    t::geometry::Image test_img = CreateTestImage();

    EXPECT_EQ(img.GetRows(), 150);
    EXPECT_EQ(img.GetCols(), 100);
    EXPECT_EQ(img.GetChannels(), 3);
    EXPECT_EQ(img.GetDtype(), core::UInt8);
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());
    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));

    EXPECT_TRUE(t::io::ReadImage(utility::GetDataPathCommon("test_imageio.jpg"),
                                 img));
    EXPECT_EQ(img.GetRows(), 150);
    EXPECT_EQ(img.GetCols(), 100);
    EXPECT_EQ(img.GetChannels(), 3);
    EXPECT_EQ(img.GetDtype(), core::UInt8);
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());
    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

TEST(ImageIO, ReadImageFromPNG) {
    WriteTestImage(CreateTestImage());
    t::geometry::Image img;
    EXPECT_TRUE(t::io::ReadImageFromPNG(
            utility::GetDataPathCommon("test_imageio.png"), img));
    t::geometry::Image test_img = CreateTestImage();

    EXPECT_EQ(img.GetRows(), 150);
    EXPECT_EQ(img.GetCols(), 100);
    EXPECT_EQ(img.GetChannels(), 3);
    EXPECT_EQ(img.GetDtype(), core::UInt8);
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

TEST(ImageIO, WriteImageToPNG) {
    WriteTestImage(CreateTestImage());
    t::geometry::Image img = CreateTestImage();
    EXPECT_TRUE(t::io::WriteImageToPNG(
            utility::GetDataPathCommon("test_imageio.png"), img));

    t::geometry::Image read_img = *(t::io::CreateImageFromFile(
            utility::GetDataPathCommon("test_imageio.png")));

    EXPECT_EQ(img.GetRows(), read_img.GetRows());
    EXPECT_EQ(img.GetCols(), read_img.GetCols());
    EXPECT_EQ(img.GetChannels(), read_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), read_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), read_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(read_img.AsTensor()));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

TEST(ImageIO, ReadImageFromJPG) {
    WriteTestImage(CreateTestImage());
    t::geometry::Image img;
    EXPECT_TRUE(t::io::ReadImageFromJPG(
            utility::GetDataPathCommon("test_imageio.jpg"), img));
    t::geometry::Image test_img = CreateTestImage();

    EXPECT_EQ(img.GetRows(), 150);
    EXPECT_EQ(img.GetCols(), 100);
    EXPECT_EQ(img.GetChannels(), 3);
    EXPECT_EQ(img.GetDtype(), core::UInt8);
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

TEST(ImageIO, WriteImageToJPG) {
    WriteTestImage(CreateTestImage());
    t::geometry::Image img = CreateTestImage();
    EXPECT_TRUE(t::io::WriteImageToJPG(
            utility::GetDataPathCommon("test_imageio.jpg"), img));

    t::geometry::Image read_img = *(t::io::CreateImageFromFile(
            utility::GetDataPathCommon("test_imageio.png")));

    EXPECT_EQ(img.GetRows(), read_img.GetRows());
    EXPECT_EQ(img.GetCols(), read_img.GetCols());
    EXPECT_EQ(img.GetChannels(), read_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), read_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), read_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(read_img.AsTensor()));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio.png"));
}

// JPG supports only UInt8, and PNG supports both UInt8 and UInt16.
// All other data types are expected to fail.
TEST(ImageIO, DifferentDtype) {
    EXPECT_TRUE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::UInt16)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::Float32)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::Float64)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::Int32)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::Int64)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 3, core::Bool)));

    EXPECT_TRUE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::UInt8)));
    EXPECT_TRUE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::UInt16)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::Float32)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::Float64)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::Int32)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::Int64)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.png"),
            t::geometry::Image(100, 200, 3, core::Bool)));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio_dtype.jpg"));
    RemoveTestImage(utility::GetDataPathCommon("test_imageio_dtype.png"));
}

TEST(ImageIO, CornerCases) {
    EXPECT_ANY_THROW(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 0, core::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 0, 3, core::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(0, 200, 3, core::UInt8)));
    EXPECT_TRUE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jpg"),
            t::geometry::Image(100, 200, 1, core::UInt8)));

    // Wrong extension
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.jg"),
            t::geometry::Image(100, 0, 3, core::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            utility::GetDataPathCommon("test_imageio_dtype.pg"),
            t::geometry::Image(100, 0, 3, core::UInt8)));

    RemoveTestImage(utility::GetDataPathCommon("test_imageio_dtype.jpg"));
}

}  // namespace tests
}  // namespace open3d
