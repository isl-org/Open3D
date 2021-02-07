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

#include "open3d/t/geometry/Image.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/TensorList.h"
#include "open3d/io/ImageIO.h"
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
    t::geometry::Image im;
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
    t::geometry::Image im(rows, cols, channels, dtype, device);
    EXPECT_EQ(im.GetRows(), rows);
    EXPECT_EQ(im.GetCols(), cols);
    EXPECT_EQ(im.GetChannels(), channels);
    EXPECT_EQ(im.GetDtype(), dtype);
    EXPECT_EQ(im.GetDevice(), device);

    // Unsupported shape or channel.
    EXPECT_ANY_THROW(t::geometry::Image(-1, cols, channels, dtype, device));
    EXPECT_ANY_THROW(t::geometry::Image(rows, -1, channels, dtype, device));
    EXPECT_ANY_THROW(t::geometry::Image(rows, cols, 0, dtype, device));
    EXPECT_ANY_THROW(t::geometry::Image(rows, cols, -1, dtype, device));

    // Check all dtypes.
    for (const core::Dtype& dtype : {
                 core::Dtype::Float32,
                 core::Dtype::Float64,
                 core::Dtype::Int32,
                 core::Dtype::Int64,
                 core::Dtype::UInt8,
                 core::Dtype::UInt16,
                 core::Dtype::Bool,
         }) {
        EXPECT_NO_THROW(
                t::geometry::Image(rows, cols, channels, dtype, device));
    }
}

TEST_P(ImagePermuteDevices, ConstructorFromTensor) {
    core::Device device = GetParam();

    int64_t rows = 480;
    int64_t cols = 640;
    int64_t channels = 3;
    core::Dtype dtype = core::Dtype::UInt8;

    // 2D Tensor. IsSame() tests memory sharing and shape matching.
    core::Tensor t_2d({rows, cols}, dtype, device);
    t::geometry::Image im_2d(t_2d);
    EXPECT_FALSE(im_2d.AsTensor().IsSame(t_2d));
    EXPECT_TRUE(im_2d.AsTensor().Reshape(t_2d.GetShape()).IsSame(t_2d));

    // 3D Tensor.
    core::Tensor t_3d({rows, cols, channels}, dtype, device);
    t::geometry::Image im_3d(t_3d);
    EXPECT_TRUE(im_3d.AsTensor().IsSame(t_3d));

    // Not 2D nor 3D.
    core::Tensor t_4d({rows, cols, channels, channels}, dtype, device);
    EXPECT_ANY_THROW(t::geometry::Image im_4d(t_4d); (void)im_4d;);

    // Non-contiguous tensor.
    // t_3d_sliced = t_3d[:, :, 0:3:2]
    core::Tensor t_3d_sliced = t_3d.Slice(2, 0, 3, 2);
    EXPECT_EQ(t_3d_sliced.GetShape(), core::SizeVector({rows, cols, 2}));
    EXPECT_FALSE(t_3d_sliced.IsContiguous());
    EXPECT_ANY_THROW(t::geometry::Image im_nc(t_3d_sliced); (void)im_nc;);
}

TEST_P(ImagePermuteDevicePairs, CopyDevice) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Tensor data =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, src_device);
    t::geometry::Image im(data);

    // Copy is created on the dst_device.
    t::geometry::Image im_copy = im.To(dst_device, /*copy=*/true);

    EXPECT_EQ(im_copy.GetDevice(), dst_device);
    EXPECT_EQ(im_copy.GetDtype(), im.GetDtype());
}

TEST_P(ImagePermuteDevices, Copy) {
    core::Device device = GetParam();

    core::Tensor data =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    t::geometry::Image im(data);

    // Copy is on the same device as source.
    t::geometry::Image im_copy = im.Clone();

    // Copy does not share the same memory with source (deep copy).
    EXPECT_FALSE(im_copy.AsTensor().IsSame(im.AsTensor()));

    // Copy has the same attributes and values as source.
    EXPECT_TRUE(im_copy.AsTensor().AllClose(im.AsTensor()));
}

// Test automatic scale determination for conversion from UInt8 / UInt16 ->
// Float32/64 and LinearTransform().
// Currently needs IPP.
TEST_P(ImagePermuteDevices,
       OPEN3D_CONCATENATE(IPP_CONDITIONAL_TEST_STR, To_LinearTransform)) {
    using ::testing::ElementsAreArray;
    using ::testing::FloatEq;
    core::Device device = GetParam();

    // reference data
    const std::vector<uint8_t> input_data = {10, 25, 0, 13};
    auto output_ref = {FloatEq(10. / 255), FloatEq(25. / 255), FloatEq(0.),
                       FloatEq(13. / 255)};
    auto negative_image_ref = {FloatEq(1. - 10. / 255), FloatEq(1. - 25. / 255),
                               FloatEq(1.), FloatEq(1. - 13. / 255)};

    t::geometry::Image input(
            core::Tensor{input_data, {2, 2, 1}, core::Dtype::UInt8, device});
    // UInt8 -> Float32: auto scale = 1./255
    t::geometry::Image output = input.To(core::Dtype::Float32);
    EXPECT_EQ(output.GetDtype(), core::Dtype::Float32);
    EXPECT_THAT(output.AsTensor().ToFlatVector<float>(),
                ElementsAreArray(output_ref));

    // LinearTransform to negative image
    output.LinearTransform(/* scale= */ -1, /* offset= */ 1);
    EXPECT_THAT(output.AsTensor().ToFlatVector<float>(),
                ElementsAreArray(negative_image_ref));

    // UInt8 -> UInt16: auto scale = 1
    output = input.To(core::Dtype::UInt16);
    EXPECT_EQ(output.GetDtype(), core::Dtype::UInt16);
    EXPECT_THAT(output.AsTensor().ToFlatVector<uint16_t>(),
                ElementsAreArray(input_data));
}

TEST_P(ImagePermuteDevices, FilterBilateral) {
    core::Device device = GetParam();

    core::Tensor data =
            core::Tensor::Zeros({10, 10}, core::Dtype::Float32, device);
    data.Slice(0, 4, 5).Slice(1, 4, 5) = core::Tensor(
            std::vector<float>{1.0f}, {}, core::Dtype::Float32, device);

    t::geometry::Image im(data.View({10, 10, 1}));
    im = im.FilterBilateral(3, 1, 1);

    utility::LogInfo("{}", im.AsTensor().View({10, 10}).ToString());
}

TEST_P(ImagePermuteDevices, FilterGaussian) {
    core::Device device = GetParam();

    core::Tensor data =
            core::Tensor::Zeros({10, 10}, core::Dtype::Float32, device);
    data.Slice(0, 4, 5).Slice(1, 4, 5) = core::Tensor(
            std::vector<float>{1.0f}, {}, core::Dtype::Float32, device);

    t::geometry::Image im(data.View({10, 10, 1}));

    // NVIDIA is not following what it is describing. Use advanced kernels
    // instead.
    im = im.FilterGaussian(5);
    utility::LogInfo("{}", im.AsTensor().View({10, 10}).ToString());
    utility::LogInfo("{}", im.AsTensor().Sum({0, 1}).ToString());

    // t::geometry::Image color = t::geometry::Image::FromLegacyImage(
    //         *io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
    //                                  "/RGBD/color/00000.jpg"),
    //         device);
    // t::geometry::Image depth = t::geometry::Image::FromLegacyImage(
    //         *io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
    //                                  "/RGBD/depth/00000.png"),
    //         device);

    // auto color_filtered = color.FilterGaussian(7);
    // auto depth_filtered = depth.FilterGaussian(7);

    // depth.AsTensor().Save("original.npy");
    // depth_filtered.AsTensor().Save("filtered.npy");

    // io::WriteImage(fmt::format("color_gaussian_{}.png", device.ToString()),
    //                color_filtered.ToLegacyImage());
    // io::WriteImage(fmt::format("depth_gaussian_{}.png", device.ToString()),
    //                depth_filtered.ToLegacyImage());
}

TEST_P(ImagePermuteDevices, FilterSobel) {
    core::Device device = GetParam();

    // clang-format off
    const std::vector<float> input_data =
      {0, 0, 0, 0, 1,
       0, 1, 1, 0, 0,
       0, 0, 1, 0, 0,
       1, 0, 1, 0, 0,
       0, 0, 1, 1, 0};
    const std::vector<float> output_dx_ref =
      {1, 1, -1, 2, 3,
       2, 3, -2, -2, 1,
       0, 3, -1, -4, 0,
       -2, 2, 1, -4, -1,
       -1, 3, 3, -4, -3};
    const std::vector<float> output_dy_ref =
      {1, 3, 3, 0, -3,
       0, 1, 2, 0, -3,
       2, -1, -1, 0, 0,
       0, 0, 1, 2, 1,
       -3, -1, 1, 2, 1};
    // clang-format on

    {  // Float32 -> Float32
        core::Tensor data = core::Tensor(input_data, {5, 5, 1},
                                         core::Dtype::Float32, device);
        t::geometry::Image im(data);
        t::geometry::Image dx, dy;
        std::tie(dx, dy) = im.FilterSobel(3);

        EXPECT_EQ(dx.GetRows(), 5);
        EXPECT_EQ(dx.GetCols(), 5);
        EXPECT_EQ(dx.GetChannels(), 1);
        EXPECT_EQ(dx.GetDtype(), core::Dtype::Float32);
        EXPECT_TRUE(dx.AsTensor().AllClose(core::Tensor(
                output_dx_ref, {5, 5, 1}, core::Dtype::Float32, device)));

        EXPECT_EQ(dy.GetRows(), 5);
        EXPECT_EQ(dy.GetCols(), 5);
        EXPECT_EQ(dy.GetChannels(), 1);
        EXPECT_EQ(dy.GetDtype(), core::Dtype::Float32);
        EXPECT_TRUE(dy.AsTensor().AllClose(core::Tensor(
                output_dy_ref, {5, 5, 1}, core::Dtype::Float32, device)));
    }

    {  // UInt8 -> Int16
        core::Tensor data = core::Tensor(input_data, {5, 5, 1},
                                         core::Dtype::Float32, device)
                                    .To(core::Dtype::UInt8);
        t::geometry::Image im(data);
        t::geometry::Image dx, dy;
        std::tie(dx, dy) = im.FilterSobel(3);

        EXPECT_EQ(dx.GetRows(), 5);
        EXPECT_EQ(dx.GetCols(), 5);
        EXPECT_EQ(dx.GetChannels(), 1);
        EXPECT_EQ(dx.GetDtype(), core::Dtype::Int16);
        EXPECT_TRUE(dx.AsTensor().AllClose(
                core::Tensor(output_dx_ref, {5, 5, 1}, core::Dtype::Float32,
                             device)
                        .To(core::Dtype::Int16)));

        EXPECT_EQ(dy.GetRows(), 5);
        EXPECT_EQ(dy.GetCols(), 5);
        EXPECT_EQ(dy.GetChannels(), 1);
        EXPECT_EQ(dy.GetDtype(), core::Dtype::Int16);
        EXPECT_TRUE(dy.AsTensor().AllClose(
                core::Tensor(output_dy_ref, {5, 5, 1}, core::Dtype::Float32,
                             device)
                        .To(core::Dtype::Int16)));
    }
}

TEST_P(ImagePermuteDevices, Resize) {
    core::Device device = GetParam();

    t::geometry::Image rgb = t::geometry::Image::FromLegacyImage(
            *io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                     "/lena_color.jpg"),
            device);
    t::geometry::Image gray = rgb.RGBToGray();

    t::geometry::Image depth = t::geometry::Image::FromLegacyImage(
            *io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                     "/RGBD/depth/00000.png"),
            device);

    rgb = rgb.Resize(2, t::geometry::Image::Nearest);
    rgb = rgb.Resize(0.5, t::geometry::Image::Lanczos);
    rgb = rgb.Resize(2, t::geometry::Image::Cubic);
    rgb = rgb.Resize(0.5, t::geometry::Image::Super);
    io::WriteImage("rgb_" + device.ToString() + ".png", rgb.ToLegacyImage());

    gray = gray.Resize(2, t::geometry::Image::Lanczos);
    gray = gray.Resize(0.5, t::geometry::Image::Nearest);
    gray = gray.Resize(2, t::geometry::Image::Linear);
    gray = gray.Resize(0.5, t::geometry::Image::Super);
    io::WriteImage("gray_" + device.ToString() + ".png", gray.ToLegacyImage());

    depth = depth.Resize(2, t::geometry::Image::Nearest);
    depth = depth.Resize(0.5, t::geometry::Image::Linear);
    depth = depth.Resize(2, t::geometry::Image::Lanczos);
    depth = depth.Resize(0.5, t::geometry::Image::Cubic);
    io::WriteImage("depth_" + device.ToString() + ".png",
                   depth.ToLegacyImage());
}

TEST_P(ImagePermuteDevices, PyrDown) {
    core::Device device = GetParam();

    t::geometry::Image rgb = t::geometry::Image::FromLegacyImage(
            *io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                     "/lena_color.jpg"),
            device);
    t::geometry::Image depth = t::geometry::Image::FromLegacyImage(
            *io::CreateImageFromFile(std::string(TEST_DATA_DIR) +
                                     "/RGBD/depth/00000.png"),
            device);

    for (int i = 0; i < 3; ++i) {
        rgb = rgb.PyrDown();
        depth = depth.PyrDown();
        io::WriteImage(
                fmt::format("rgb_pyr_{:03d}_{}.png", i, device.ToString()),
                rgb.ToLegacyImage());
        io::WriteImage(
                fmt::format("depth_pyr_{:03d}_{}.png", i, device.ToString()),
                depth.ToLegacyImage());
    }
}

TEST_P(ImagePermuteDevices, Dilate) {
    using ::testing::ElementsAreArray;

    // reference data used to validate the filtering of an image
    // clang-format off
    const std::vector<float> input_data = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1.2, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<float> output_ref = {
        1.2, 1.2, 1, 0, 0, 1, 1, 1,
        1.2, 1.2, 1, 1, 0, 1, 1, 1,
        1.2, 1.2, 1, 1, 0, 1, 1, 1,
        0, 1, 1, 1, 0, 0, 0, 0};
    // clang-format on

    // test image dimensions
    const int rows = 4;
    const int cols = 8;
    const int channels = 1;
    const int kernel_size = 3;
    core::Device device = GetParam();

    core::Tensor t_input{
            input_data, {rows, cols, channels}, core::Dtype::Float32, device};
    t::geometry::Image input(t_input);
    t::geometry::Image output;

    // UInt8
    core::Tensor t_input_uint8_t =
            t_input.To(core::Dtype::UInt8);  // normal static_cast is OK
    t::geometry::Image input_uint8_t(t_input_uint8_t);
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {  // Not Implemented
        ASSERT_THROW(input_uint8_t.Dilate(kernel_size), std::runtime_error);
    } else {
        output = input_uint8_t.Dilate(kernel_size);
        EXPECT_EQ(output.GetRows(), input.GetRows());
        EXPECT_EQ(output.GetCols(), input.GetCols());
        EXPECT_EQ(output.GetChannels(), input.GetChannels());
        EXPECT_THAT(output.AsTensor().ToFlatVector<uint8_t>(),
                    ElementsAreArray(output_ref));
    }

    // UInt16
    core::Tensor t_input_uint16_t =
            t_input.To(core::Dtype::UInt16);  // normal static_cast is OK
    t::geometry::Image input_uint16_t(t_input_uint16_t);
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {  // Not Implemented
        ASSERT_THROW(input_uint16_t.Dilate(kernel_size), std::runtime_error);
    } else {
        output = input_uint16_t.Dilate(kernel_size);
        EXPECT_EQ(output.GetRows(), input.GetRows());
        EXPECT_EQ(output.GetCols(), input.GetCols());
        EXPECT_EQ(output.GetChannels(), input.GetChannels());
        EXPECT_THAT(output.AsTensor().ToFlatVector<uint16_t>(),
                    ElementsAreArray(output_ref));
    }

    // Float32
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {  // Not Implemented
        ASSERT_THROW(input.Dilate(kernel_size), std::runtime_error);
    } else {
        output = input.Dilate(kernel_size);
        EXPECT_EQ(output.GetRows(), input.GetRows());
        EXPECT_EQ(output.GetCols(), input.GetCols());
        EXPECT_EQ(output.GetChannels(), input.GetChannels());
        EXPECT_THAT(output.AsTensor().ToFlatVector<float>(),
                    ElementsAreArray(output_ref));
    }
}

// tImage: (r, c, ch) | legacy Image: (u, v, ch) = (c, r, ch)
TEST_P(ImagePermuteDevices, ToLegacyImage) {
    core::Device device = GetParam();
    // 2 byte dtype is general enough for uin8_t as well as float
    core::Dtype dtype = core::Dtype::UInt16;

    // 2D tensor for 1 channel image
    core::Tensor t_1ch(std::vector<uint16_t>{0, 1, 2, 3, 4, 5}, {2, 3}, dtype,
                       device);

    // Test 1 channel image conversion
    t::geometry::Image im_1ch(t_1ch);
    geometry::Image leg_im_1ch = im_1ch.ToLegacyImage();
    for (int r = 0; r < im_1ch.GetRows(); ++r)
        for (int c = 0; c < im_1ch.GetCols(); ++c)
            EXPECT_EQ(im_1ch.At(r, c).Item<uint16_t>(),
                      *leg_im_1ch.PointerAt<uint16_t>(c, r));

    // 3D tensor for 3 channel image
    core::Tensor t_3ch(
            std::vector<uint16_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            {2, 2, 3}, dtype, device);
    // Test 3 channel image conversion
    t::geometry::Image im_3ch(t_3ch);
    geometry::Image leg_im_3ch = im_3ch.ToLegacyImage();
    for (int r = 0; r < im_3ch.GetRows(); ++r)
        for (int c = 0; c < im_3ch.GetCols(); ++c)
            for (int ch = 0; ch < im_3ch.GetChannels(); ++ch)
                EXPECT_EQ(im_3ch.At(r, c, ch).Item<uint16_t>(),
                          *leg_im_3ch.PointerAt<uint16_t>(c, r, ch));
}

}  // namespace tests
}  // namespace open3d
