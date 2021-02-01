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

#include "open3d/t/io/ImageIO.h"

#include <gtest/gtest.h>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"
#include "open3d/t/geometry/Image.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

namespace {

t::geometry::Image CreateTestImage() {
    core::Tensor t(core::SizeVector{150, 100, 3}, core::Dtype::UInt8);
    t.Slice(2, 0, 1).Fill(250);
    t.Slice(2, 1, 2).Fill(150);
    t.Slice(2, 2, 3).Fill(200);

    auto test_img = t::geometry::Image(t);

    return test_img;
}
}  // namespace

// Write test image.
TEST(ImageIO, WriteImage) {
    auto test_img = CreateTestImage();
    EXPECT_TRUE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio.png", test_img));
    EXPECT_TRUE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio.jpg", test_img));
}

TEST(ImageIO, CreateImageFromFile) {
    auto img_png = t::io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                              "/test_imageio.png");
    auto img_jpg = t::io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                              "/test_imageio.jpg");

    EXPECT_EQ(img_png->GetRows(), img_jpg->GetRows());
    EXPECT_EQ(img_png->GetCols(), img_jpg->GetCols());
    EXPECT_EQ(img_png->GetChannels(), img_jpg->GetChannels());
    EXPECT_EQ(img_png->GetDtype(), img_jpg->GetDtype());
    EXPECT_EQ(img_png->GetDevice(), img_jpg->GetDevice());

    auto test_img = CreateTestImage();

    EXPECT_EQ(img_png->GetRows(), test_img.GetRows());
    EXPECT_EQ(img_png->GetCols(), test_img.GetCols());
    EXPECT_EQ(img_png->GetChannels(), test_img.GetChannels());
    EXPECT_EQ(img_png->GetDtype(), test_img.GetDtype());
    EXPECT_EQ(img_png->GetDevice(), test_img.GetDevice());

    EXPECT_FALSE(img_jpg->AsTensor().IsSame(test_img.AsTensor()));
    EXPECT_TRUE(img_jpg->AsTensor().AllClose(test_img.AsTensor()));
    EXPECT_TRUE(img_png->AsTensor().AllClose(test_img.AsTensor()));
}

TEST(ImageIO, ReadImage) {
    t::geometry::Image img;
    EXPECT_TRUE(t::io::ReadImage(
            std::string(TEST_DATA_DIR) + "/test_imageio.png", img));
    auto test_img = CreateTestImage();

    EXPECT_EQ(img.GetRows(), test_img.GetRows());
    EXPECT_EQ(img.GetCols(), test_img.GetCols());
    EXPECT_EQ(img.GetChannels(), test_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), test_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());
    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));

    EXPECT_TRUE(t::io::ReadImage(
            std::string(TEST_DATA_DIR) + "/test_imageio.jpg", img));
    EXPECT_EQ(img.GetRows(), test_img.GetRows());
    EXPECT_EQ(img.GetCols(), test_img.GetCols());
    EXPECT_EQ(img.GetChannels(), test_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), test_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());
    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));
}

TEST(ImageIO, ReadImageFromPNG) {
    t::geometry::Image img;
    EXPECT_TRUE(t::io::ReadImageFromPNG(
            std::string(TEST_DATA_DIR) + "/test_imageio.png", img));
    auto test_img = CreateTestImage();

    EXPECT_EQ(img.GetRows(), test_img.GetRows());
    EXPECT_EQ(img.GetCols(), test_img.GetCols());
    EXPECT_EQ(img.GetChannels(), test_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), test_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));
}

TEST(ImageIO, WriteImageToPNG) {
    auto img = CreateTestImage();
    EXPECT_TRUE(t::io::WriteImageToPNG(
            std::string(TEST_DATA_DIR) + "/test_imageio.png", img));

    auto read_img = *(t::io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                                 "/test_imageio.png"));

    EXPECT_EQ(img.GetRows(), read_img.GetRows());
    EXPECT_EQ(img.GetCols(), read_img.GetCols());
    EXPECT_EQ(img.GetChannels(), read_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), read_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), read_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(read_img.AsTensor()));
}

TEST(ImageIO, ReadImageFromJPG) {
    t::geometry::Image img;
    EXPECT_TRUE(t::io::ReadImageFromJPG(
            std::string(TEST_DATA_DIR) + "/test_imageio.jpg", img));
    auto test_img = CreateTestImage();

    EXPECT_EQ(img.GetRows(), test_img.GetRows());
    EXPECT_EQ(img.GetCols(), test_img.GetCols());
    EXPECT_EQ(img.GetChannels(), test_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), test_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), test_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(test_img.AsTensor()));
}

TEST(ImageIO, WriteImageToJPG) {
    auto img = CreateTestImage();
    EXPECT_TRUE(t::io::WriteImageToJPG(
            std::string(TEST_DATA_DIR) + "/test_imageio.jpg", img));

    auto read_img = *(t::io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                                 "/test_imageio.png"));

    EXPECT_EQ(img.GetRows(), read_img.GetRows());
    EXPECT_EQ(img.GetCols(), read_img.GetCols());
    EXPECT_EQ(img.GetChannels(), read_img.GetChannels());
    EXPECT_EQ(img.GetDtype(), read_img.GetDtype());
    EXPECT_EQ(img.GetDevice(), read_img.GetDevice());

    EXPECT_TRUE(img.AsTensor().AllClose(read_img.AsTensor()));
}

TEST(ImageIO, DifferentDtype) {
    EXPECT_TRUE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::UInt16)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::Float32)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::Float64)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::Int32)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::Int64)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 3, core::Dtype::Bool)));

    EXPECT_TRUE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::UInt8)));
    EXPECT_TRUE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::UInt16)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::Float32)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::Float64)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::Int32)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::Int64)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.png",
            t::geometry::Image(100, 200, 3, core::Dtype::Bool)));
}

TEST(ImageIO, CornerCases) {
    EXPECT_ANY_THROW(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 0, core::Dtype::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 0, 3, core::Dtype::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(0, 200, 3, core::Dtype::UInt8)));
    EXPECT_TRUE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jpg",
            t::geometry::Image(100, 200, 1, core::Dtype::UInt8)));

    // Wrong extension
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.jg",
            t::geometry::Image(100, 0, 3, core::Dtype::UInt8)));
    EXPECT_FALSE(t::io::WriteImage(
            std::string(TEST_DATA_DIR) + "/test_imageio_dtype.pg",
            t::geometry::Image(100, 0, 3, core::Dtype::UInt8)));
}

}  // namespace tests
}  // namespace open3d
