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

#include "open3d/geometry/RGBDImage.h"

#include <vector>

#include "open3d/geometry/Image.h"
#include "open3d/io/ImageIO.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(RGBDImage, Constructor) {
    geometry::Image image;
    geometry::Image color;

    const int size = 5;

    // test image dimensions
    const int image_width = size;
    const int image_height = size;
    const int image_num_of_channels = 1;
    const int image_bytes_per_channel = 1;

    const int color_width = size;
    const int color_height = size;
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    color.Prepare(color_width, color_height, color_num_of_channels,
                  color_bytes_per_channel);

    image.Prepare(image_width, image_height, image_num_of_channels,
                  image_bytes_per_channel);

    Rand(image.data_, 100, 150, 0);
    Rand(color.data_, 130, 200, 0);

    auto depth = image.ConvertDepthToFloatImage();

    geometry::RGBDImage rgbd_image(color, *depth);

    ExpectEQ(color.data_, rgbd_image.color_.data_);
    ExpectEQ(depth->data_, rgbd_image.depth_.data_);
}

TEST(RGBDImage, DISABLED_MemberData) { NotImplemented(); }

std::pair<float, float> FloatImageMinMax(const geometry::Image& im) {
    if (im.bytes_per_channel_ != 4) {
        utility::LogError("im must be a float image.");
    }
    if (im.width_ * im.height_ == 0) {
        return std::make_pair(0.0, 0.0);
    }
    const float* float_data = reinterpret_cast<const float*>(im.data_.data());
    float min_val = float_data[0];
    float max_val = float_data[0];
    for (int i = 0; i < im.width_ * im.height_ * im.num_of_channels_; i++) {
        min_val = std::min(float_data[i], min_val);
        max_val = std::max(float_data[i], max_val);
    }
    return std::make_pair(min_val, max_val);
}

TEST(RGBDImage, CreateFromColorAndDepth) {
    geometry::Image im_color;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/color/00000.jpg", im_color));
    EXPECT_EQ(im_color.num_of_channels_, 3);
    EXPECT_EQ(im_color.bytes_per_channel_, 1);

    geometry::Image im_depth;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00000.png", im_depth));
    EXPECT_EQ(im_depth.num_of_channels_, 1);
    EXPECT_EQ(im_depth.bytes_per_channel_, 2);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth);
    EXPECT_EQ(im_rgbd->color_.width_, 640);
    EXPECT_EQ(im_rgbd->color_.height_, 480);
    EXPECT_EQ(im_rgbd->color_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->color_.bytes_per_channel_, 4);
    EXPECT_EQ(im_rgbd->depth_.width_, 640);
    EXPECT_EQ(im_rgbd->depth_.height_, 480);
    EXPECT_EQ(im_rgbd->depth_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->depth_.bytes_per_channel_, 4);

    // Check data scale and truncation. These values are determined by inputs.
    float min_val;
    float max_val;
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->color_);
    EXPECT_FLOAT_EQ(min_val, 0.008207843);
    EXPECT_FLOAT_EQ(max_val, 1.0);
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->depth_);
    EXPECT_FLOAT_EQ(min_val, 0.0);
    EXPECT_FLOAT_EQ(max_val, 2.702);
}

TEST(RGBDImage, CreateFromRedwoodFormat) {
    geometry::Image im_color;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/color/00000.jpg", im_color));
    EXPECT_EQ(im_color.num_of_channels_, 3);
    EXPECT_EQ(im_color.bytes_per_channel_, 1);

    geometry::Image im_depth;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00000.png", im_depth));
    EXPECT_EQ(im_depth.num_of_channels_, 1);
    EXPECT_EQ(im_depth.bytes_per_channel_, 2);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromRedwoodFormat(im_color, im_depth);
    EXPECT_EQ(im_rgbd->color_.width_, 640);
    EXPECT_EQ(im_rgbd->color_.height_, 480);
    EXPECT_EQ(im_rgbd->color_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->color_.bytes_per_channel_, 4);
    EXPECT_EQ(im_rgbd->depth_.width_, 640);
    EXPECT_EQ(im_rgbd->depth_.height_, 480);
    EXPECT_EQ(im_rgbd->depth_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->depth_.bytes_per_channel_, 4);

    // Check data scale and truncation. These values are determined by inputs.
    float min_val;
    float max_val;
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->color_);
    EXPECT_FLOAT_EQ(min_val, 0.008207843);
    EXPECT_FLOAT_EQ(max_val, 1.0);
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->depth_);
    EXPECT_FLOAT_EQ(min_val, 0.0);
    EXPECT_FLOAT_EQ(max_val, 2.702);
}

TEST(RGBDImage, CreateFromTUMFormat) {
    geometry::Image im_color;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/other_formats/TUM_color.png",
            im_color));
    EXPECT_EQ(im_color.num_of_channels_, 3);
    EXPECT_EQ(im_color.bytes_per_channel_, 1);

    geometry::Image im_depth;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/other_formats/TUM_depth.png",
            im_depth));
    EXPECT_EQ(im_depth.num_of_channels_, 1);
    EXPECT_EQ(im_depth.bytes_per_channel_, 2);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromTUMFormat(im_color, im_depth);
    EXPECT_EQ(im_rgbd->color_.width_, 640);
    EXPECT_EQ(im_rgbd->color_.height_, 480);
    EXPECT_EQ(im_rgbd->color_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->color_.bytes_per_channel_, 4);
    EXPECT_EQ(im_rgbd->depth_.width_, 640);
    EXPECT_EQ(im_rgbd->depth_.height_, 480);
    EXPECT_EQ(im_rgbd->depth_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->depth_.bytes_per_channel_, 4);

    // Check data scale and truncation. These values are determined by inputs.
    float min_val;
    float max_val;
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->color_);
    EXPECT_FLOAT_EQ(min_val, 0.0);
    EXPECT_FLOAT_EQ(max_val, 0.99748623);
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->depth_);
    EXPECT_FLOAT_EQ(min_val, 0.0);
    EXPECT_FLOAT_EQ(max_val, 3.994);
}

TEST(RGBDImage, CreateFromSUNFormat) {
    geometry::Image im_color;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/other_formats/SUN_color.jpg",
            im_color));
    EXPECT_EQ(im_color.num_of_channels_, 3);
    EXPECT_EQ(im_color.bytes_per_channel_, 1);

    geometry::Image im_depth;
    EXPECT_TRUE(io::ReadImage(
            std::string(TEST_DATA_DIR) + "/RGBD/other_formats/SUN_depth.png",
            im_depth));
    EXPECT_EQ(im_depth.num_of_channels_, 1);
    EXPECT_EQ(im_depth.bytes_per_channel_, 2);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromSUNFormat(im_color, im_depth);
    EXPECT_EQ(im_rgbd->color_.width_, 640);
    EXPECT_EQ(im_rgbd->color_.height_, 480);
    EXPECT_EQ(im_rgbd->color_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->color_.bytes_per_channel_, 4);
    EXPECT_EQ(im_rgbd->depth_.width_, 640);
    EXPECT_EQ(im_rgbd->depth_.height_, 480);
    EXPECT_EQ(im_rgbd->depth_.num_of_channels_, 1);
    EXPECT_EQ(im_rgbd->depth_.bytes_per_channel_, 4);

    // Check data scale and truncation. These values are determined by inputs.
    float min_val;
    float max_val;
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->color_);
    EXPECT_FLOAT_EQ(min_val, 0.0);
    EXPECT_FLOAT_EQ(max_val, 1.0);
    std::tie(min_val, max_val) = FloatImageMinMax(im_rgbd->depth_);
    EXPECT_FLOAT_EQ(min_val, 0.0);
    EXPECT_FLOAT_EQ(max_val, 6.889);
}

TEST(RGBDImage, CreateFromNYUFormat) {
    GTEST_SKIP() << "NYU dataset is in .ppm and .pgm format and needs "
                    "matplotlib's mpimg reader. CreateFromNYUFormat is similar "
                    "to other RGBD dataset loader, just with a different "
                    "scaling and truncation factor.";
}

template <typename T>
static std::vector<T> ImageAsVector(const geometry::Image& im) {
    const T* data_ptr = reinterpret_cast<const T*>(im.data_.data());
    int num_elements = im.height_ * im.width_ * im.num_of_channels_;
    std::vector<T> vals(data_ptr, data_ptr + num_elements);
    return vals;
}

TEST(RGBDImage, FilterPyramid) {
    int width = 4;
    int height = 4;

    geometry::Image im_color;
    im_color.Prepare(width, height, 3, 4);
    float* im_color_data = reinterpret_cast<float*>(im_color.data_.data());
    std::vector<float> im_color_val{
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3,
            0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7,
            0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1,
            1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.4, 1.4, 1.4, 1.5, 1.5, 1.5};
    std::copy(im_color_val.begin(), im_color_val.end(), im_color_data);

    geometry::Image im_depth;
    im_depth.Prepare(width, height, 1, 2);
    uint16_t* im_depth_data =
            reinterpret_cast<uint16_t*>(im_depth.data_.data());
    std::vector<uint16_t> im_depth_val{0, 1, 2,  3,  4,  5,  6,  7,
                                       8, 9, 10, 11, 12, 13, 14, 15};
    std::copy(im_depth_val.begin(), im_depth_val.end(), im_depth_data);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth, 1,
                                                         1000, true);
    geometry::RGBDImagePyramid pyramid = im_rgbd->CreatePyramid(2, false);
    geometry::RGBDImagePyramid pyramid_filtered =
            geometry::RGBDImage::FilterPyramid(
                    pyramid, geometry::Image::FilterType::Sobel3Dx);

    ExpectEQ(ImageAsVector<float>(pyramid_filtered[0]->color_),
             std::vector<float>({0.4, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.4, 0.4,
                                 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.4}));
    ExpectEQ(ImageAsVector<float>(pyramid_filtered[0]->depth_),
             std::vector<float>(
                     {4, 8, 8, 4, 4, 8, 8, 4, 4, 8, 8, 4, 4, 8, 8, 4}));
    ExpectEQ(ImageAsVector<float>(pyramid_filtered[1]->color_),
             std::vector<float>({0.8, 0.8, 0.8, 0.8}));
    ExpectEQ(ImageAsVector<float>(pyramid_filtered[1]->depth_),
             std::vector<float>({8, 8, 8, 8}));
}

TEST(RGBDImage, CreatePyramid) {
    int width = 4;
    int height = 4;

    geometry::Image im_color;
    im_color.Prepare(width, height, 3, 4);
    float* im_color_data = reinterpret_cast<float*>(im_color.data_.data());
    std::vector<float> im_color_val{
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3,
            0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7,
            0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1,
            1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 1.4, 1.4, 1.4, 1.5, 1.5, 1.5};
    std::copy(im_color_val.begin(), im_color_val.end(), im_color_data);

    geometry::Image im_depth;
    im_depth.Prepare(width, height, 1, 2);
    uint16_t* im_depth_data =
            reinterpret_cast<uint16_t*>(im_depth.data_.data());
    std::vector<uint16_t> im_depth_val{0, 1, 2,  3,  4,  5,  6,  7,
                                       8, 9, 10, 11, 12, 13, 14, 15};
    std::copy(im_depth_val.begin(), im_depth_val.end(), im_depth_data);

    std::shared_ptr<geometry::RGBDImage> im_rgbd =
            geometry::RGBDImage::CreateFromColorAndDepth(im_color, im_depth, 1,
                                                         1000, true);
    geometry::RGBDImagePyramid pyramid = im_rgbd->CreatePyramid(2, false);

    ExpectEQ(ImageAsVector<float>(pyramid[0]->color_),
             std::vector<float>({0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}));
    ExpectEQ(ImageAsVector<float>(pyramid[0]->depth_),
             std::vector<float>(
                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
    ExpectEQ(ImageAsVector<float>(pyramid[1]->color_),
             std::vector<float>({0.25, 0.45, 1.05, 1.25}));
    ExpectEQ(ImageAsVector<float>(pyramid[1]->depth_),
             std::vector<float>({2.5, 4.5, 10.5, 12.5}));
}

}  // namespace tests
}  // namespace open3d
