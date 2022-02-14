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

#include "open3d/t/geometry/Image.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/data/Dataset.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Preprocessor.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

static core::Tensor CreateIntrinsics(float down_factor = 1.0f) {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();

    return core::Tensor(
            std::vector<double>({(focal_length.first / down_factor), 0,
                                 (principal_point.first / down_factor), 0,
                                 (focal_length.second / down_factor),
                                 (principal_point.second / down_factor), 0, 0,
                                 1}),
            {3, 3}, core::Float64);
}

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
    EXPECT_EQ(im.GetDtype(), core::Float32);
    EXPECT_EQ(im.GetDevice(), core::Device("CPU:0"));
}

TEST_P(ImagePermuteDevices, Constructor) {
    core::Device device = GetParam();

    // Normal case.
    int64_t rows = 480;
    int64_t cols = 640;
    int64_t channels = 3;
    core::Dtype dtype = core::UInt8;
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
                 core::Float32,
                 core::Float64,
                 core::Int32,
                 core::Int64,
                 core::UInt8,
                 core::UInt16,
                 core::Bool,
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
    core::Dtype dtype = core::UInt8;

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

    core::Tensor data = core::Tensor::Ones({2, 3}, core::Float32, src_device);
    t::geometry::Image im(data);

    // Copy is created on the dst_device.
    t::geometry::Image im_copy = im.To(dst_device, /*copy=*/true);

    EXPECT_EQ(im_copy.GetDevice(), dst_device);
    EXPECT_EQ(im_copy.GetDtype(), im.GetDtype());
}

TEST_P(ImagePermuteDevices, Copy) {
    core::Device device = GetParam();

    core::Tensor data = core::Tensor::Ones({2, 3}, core::Float32, device);
    t::geometry::Image im(data);

    // Copy is on the same device as source.
    t::geometry::Image im_copy = im.Clone();

    // Copy does not share the same memory with source (deep copy).
    EXPECT_FALSE(im_copy.AsTensor().IsSame(im.AsTensor()));

    // Copy has the same attributes and values as source.
    EXPECT_TRUE(im_copy.AsTensor().AllClose(im.AsTensor()));
}

// a. Automatic scale determination for conversion from UInt8 / UInt16 ->
// Float32/64
// b. LinearTransform() with value saturation.
// c. 1 channel and 3 channels for all cases.
TEST_P(ImagePermuteDevices, To_LinearTransform) {
    using ::testing::ElementsAreArray;
    using ::testing::FloatEq;
    using ::testing::FloatNear;
    core::Device device = GetParam();

    // reference data
    const std::vector<uint8_t> input_data = {10, 25, 0, 13, 5, 40};
    auto output_ref = {FloatEq(10. / 255),  FloatEq(25. / 255),
                       FloatNear(0., 1e-8), FloatEq(13. / 255),
                       FloatEq(5. / 255),   FloatEq(40. / 255)};
    auto negative_image_ref = {FloatEq(1. - 10. / 255),
                               FloatEq(1. - 25. / 255),
                               FloatEq(1.),
                               FloatEq(1. - 13. / 255),
                               FloatEq(1. - 5. / 255),
                               FloatEq(1. - 40. / 255)

    };
    auto saturate_ref = {180, 255, 0, 240, 80, 255};
    core::Tensor t_input{input_data, {2, 3, 1}, core::UInt8, device};
    core::Tensor t_input3 = t_input.Broadcast({2, 3, 3}).Clone();

    t::geometry::Image input(t_input);
    // UInt8 -> Float32: auto scale = 1./255
    t::geometry::Image output = input.To(core::Float32);
    EXPECT_EQ(output.GetDtype(), core::Float32);
    EXPECT_THAT(output.AsTensor().ToFlatVector<float>(),
                ElementsAreArray(output_ref));
    // 3 channels
    t::geometry::Image input3(t_input3);
    t::geometry::Image output3 = input3.To(core::Float32);
    for (int64_t ch = 0; ch < 3; ++ch) {
        EXPECT_THAT(
                output3.AsTensor().Slice(2, ch, ch + 1).ToFlatVector<float>(),
                ElementsAreArray(output_ref));
    }

    // LinearTransform to negative image
    output.LinearTransform(/* scale= */ -1, /* offset= */ 1);
    EXPECT_THAT(output.AsTensor().ToFlatVector<float>(),
                ElementsAreArray(negative_image_ref));
    // 3 channels
    output3.LinearTransform(/* scale= */ -1, /* offset= */ 1);
    for (int64_t ch = 0; ch < 3; ++ch) {
        EXPECT_THAT(
                output3.AsTensor().Slice(2, ch, ch + 1).ToFlatVector<float>(),
                ElementsAreArray(negative_image_ref));
    }

    // UInt8 -> UInt16: auto scale = 1
    output = input.To(core::UInt16);
    EXPECT_EQ(output.GetDtype(), core::UInt16);
    EXPECT_THAT(output.AsTensor().ToFlatVector<uint16_t>(),
                ElementsAreArray(input_data));
    // 3 channels
    output3 = input3.To(core::UInt16);
    for (int64_t ch = 0; ch < 3; ++ch) {
        EXPECT_THAT(output3.AsTensor()
                            .Slice(2, ch, ch + 1)
                            .ToFlatVector<uint16_t>(),
                    ElementsAreArray(input_data));
    }

    // Saturation to [0, 255]
    output = input.LinearTransform(/* scale= */ 20, /* offset= */ -20);
    EXPECT_THAT(output.AsTensor().ToFlatVector<uint8_t>(),
                ElementsAreArray(saturate_ref));
    // 3 channels
    output3 = input3.LinearTransform(/* scale= */ 20, /* offset= */ -20);
    for (int64_t ch = 0; ch < 3; ++ch) {
        EXPECT_THAT(
                output3.AsTensor().Slice(2, ch, ch + 1).ToFlatVector<uint8_t>(),
                ElementsAreArray(saturate_ref));
    }
}

TEST_P(ImagePermuteDevices, FilterBilateral) {
    core::Device device = GetParam();

    {  // Float32
        // clang-format off
        const std::vector<float> input_data =
          {0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0};
        const std::vector<float> output_ref_ipp =
          {0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.199001, 0.0, 0.0,
           0.0, 0.199001, 0.201605, 0.199001, 0.0,
           0.0, 0.0, 0.199001, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0};
        const std::vector<float> output_ref_npp =
          {0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.110249, 0.110802, 0.110249, 0.0,
           0.0, 0.110802, 0.112351, 0.110802, 0.0,
           0.0, 0.110249, 0.110802, 0.110249, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::Float32, device);

        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.FilterBilateral(3, 10, 10), std::runtime_error);
        } else {
            im = im.FilterBilateral(3, 10, 10);
            if (device.GetType() == core::Device::DeviceType::CPU) {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_ipp, {5, 5, 1}, core::Float32, device)));
            } else {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_npp, {5, 5, 1}, core::Float32, device)));
            }
        }
    }

    {  // UInt8
        // clang-format off
        const std::vector<uint8_t> input_data =
          {0, 0, 0, 0, 0,
           0, 121, 121, 121, 0,
           0, 125, 128, 125, 0,
           0, 121, 121, 121, 0,
           0, 0, 0, 0, 0};
        const std::vector<uint8_t> output_ref_ipp =
          {0, 0, 0, 0, 0,
           0, 122, 122, 122, 0,
           0, 124, 125, 124, 0,
           0, 122, 122, 122, 0,
           0, 0, 0, 0, 0};
        const std::vector<uint8_t> output_ref_npp =
          {0, 0, 0, 0, 0,
           0, 122, 122, 122, 0,
           0, 123, 123, 123, 0,
           0, 122, 122, 122, 0,
           0, 0, 0, 0, 0};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::UInt8, device);

        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.FilterBilateral(3, 5, 5), std::runtime_error);
        } else {
            im = im.FilterBilateral(3, 5, 5);
            if (device.GetType() == core::Device::DeviceType::CPU) {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_ipp, {5, 5, 1}, core::UInt8, device)));
            } else {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_npp, {5, 5, 1}, core::UInt8, device)));
            }
        }
    }
}

// IPP and NPP are consistent when kernel_size = 3x3.
// Note: in 5 x 5 NPP adds a weird offset.
TEST_P(ImagePermuteDevices, FilterGaussian) {
    core::Device device = GetParam();

    {  // Float32
        // clang-format off
        const std::vector<float> input_data =
          {0, 0, 0, 0, 0,
           0, 1, 0, 0, 1,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 1, 0};
        const std::vector<float> output_ref =
          {0.0751136, 0.123841, 0.0751136, 0.0751136, 0.198955,
           0.123841, 0.204180, 0.123841, 0.123841, 0.328021,
           0.0751136, 0.123841, 0.0751136, 0.0751136, 0.198955,
           0.0, 0.0, 0.0751136, 0.123841, 0.0751136,
           0.0, 0.0, 0.198955, 0.328021, 0.198955};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::Float32, device);
        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.FilterGaussian(3), std::runtime_error);
        } else {
            im = im.FilterGaussian(3);
            EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                    output_ref, {5, 5, 1}, core::Float32, device)));
        }
    }

    {  // UInt8
        // clang-format off
        const std::vector<uint8_t> input_data =
          {0, 0, 0, 0, 0,
           0, 128, 0, 0, 255,
           0, 0, 0, 128, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 255, 0};
        const std::vector<uint8_t> output_ref_ipp =
          {10, 16, 10, 19, 51,
           16, 26, 25, 47, 93,
           10, 16, 25, 45, 67,
           0, 0, 29, 47, 29,
           0, 0, 51, 84, 51};
        const std::vector<uint8_t> output_ref_npp =
          {9, 15, 9, 19, 50,
           15, 26, 25, 47, 93,
           9, 15, 25, 45, 66,
           0, 0, 28, 47, 28,
           0, 0, 50, 83, 50};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::UInt8, device);
        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.FilterGaussian(3), std::runtime_error);
        } else {
            im = im.FilterGaussian(3);
            if (device.GetType() == core::Device::DeviceType::CPU) {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_ipp, {5, 5, 1}, core::UInt8, device)));
            } else {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_npp, {5, 5, 1}, core::UInt8, device)));
            }
        }
    }
}

TEST_P(ImagePermuteDevices, Filter) {
    core::Device device = GetParam();

    {  // Float32
        // clang-format off
        const std::vector<float> input_data =
          {0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 0};
       const std::vector<float> kernel_data =
          {0.00296902, 0.0133062 , 0.02193824, 0.0133062 , 1.00296902,
           0.0133062 , 0.05963413, 0.09832021, 0.05963413, 0.0133062 ,
           0.02193824, 0.09832021, 0.16210286, 0.09832021, 0.02193824,
           0.0133062 , 0.05963413, 0.09832021, 0.05963413, 0.0133062 ,
           0.00296902, 0.0133062 , 0.02193824, 0.0133062 , -1.00296902
        };
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::Float32, device);
        core::Tensor kernel =
                core::Tensor(kernel_data, {5, 5}, core::Float32, device);
        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.Filter(kernel), std::runtime_error);
        } else {
            t::geometry::Image im_new = im.Filter(kernel);
            EXPECT_TRUE(
                    im_new.AsTensor().Reverse().View({5, 5}).AllClose(kernel));
        }
    }

    {  // UInt8
        // clang-format off
        const std::vector<uint8_t> input_data =
          {0, 0, 0, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 128, 0, 0,
           0, 0, 0, 0, 0,
           0, 0, 0, 0, 255};
       const std::vector<float> kernel_data =
          {0.00296902, 0.0133062 , 0.02193824, 0.0133062 , 1.00296902,
           0.0133062 , 0.05963413, 0.09832021, 0.05963413, 0.0133062 ,
           0.02193824, 0.09832021, 0.16210286, 0.09832021, 0.02193824,
           0.0133062 , 0.05963413, 0.09832021, 0.05963413, 0.0133062 ,
           0.00296902, 0.0133062 , 0.02193824, 0.0133062 , -1.00296902
        };

       const std::vector<uint8_t> output_ref_ipp =
         {0, 2, 3, 2, 0,
          2, 8, 13, 8, 2,
          3, 13, 0, 0, 0,
          2, 8, 0, 0, 0,
          128, 2, 0, 0, 0
         };
       const std::vector<uint8_t> output_ref_npp =
         {0, 1, 2, 1, 0,
          1, 7, 12, 7, 1,
          2, 12, 0, 0, 0,
          1, 7, 0, 0, 0,
          128, 1, 0, 0, 0
         };
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::UInt8, device);
        core::Tensor kernel =
                core::Tensor(kernel_data, {5, 5}, core::Float32, device);
        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.Filter(kernel), std::runtime_error);
        } else {
            im = im.Filter(kernel);
            if (device.GetType() == core::Device::DeviceType::CPU) {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_ipp, {5, 5, 1}, core::UInt8, device)));
            } else {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_npp, {5, 5, 1}, core::UInt8, device)));
            }
        }
    }
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
        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::Float32, device);
        t::geometry::Image im(data);
        t::geometry::Image dx, dy;
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.FilterSobel(3), std::runtime_error);
        } else {
            std::tie(dx, dy) = im.FilterSobel(3);

            EXPECT_TRUE(dx.AsTensor().AllClose(core::Tensor(
                    output_dx_ref, {5, 5, 1}, core::Float32, device)));
            EXPECT_TRUE(dy.AsTensor().AllClose(core::Tensor(
                    output_dy_ref, {5, 5, 1}, core::Float32, device)));
        }
    }

    {  // UInt8 -> Int16
        core::Tensor data =
                core::Tensor(input_data, {5, 5, 1}, core::Float32, device)
                        .To(core::UInt8);
        t::geometry::Image im(data);
        t::geometry::Image dx, dy;
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.FilterSobel(3), std::runtime_error);
        } else {
            std::tie(dx, dy) = im.FilterSobel(3);

            EXPECT_TRUE(dx.AsTensor().AllClose(
                    core::Tensor(output_dx_ref, {5, 5, 1}, core::Float32,
                                 device)
                            .To(core::Int16)));
            EXPECT_TRUE(dy.AsTensor().AllClose(
                    core::Tensor(output_dy_ref, {5, 5, 1}, core::Float32,
                                 device)
                            .To(core::Int16)));
        }
    }
}

TEST_P(ImagePermuteDevices, Resize) {
    core::Device device = GetParam();

    {  // Float32
        // clang-format off
        const std::vector<float> input_data =
          {0, 0, 1, 1, 1, 1,
           0, 1, 1, 0, 0, 1,
           1, 0, 0, 1, 0, 1,
           0, 1, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1};
        const std::vector<float> output_ref =
          {0, 1, 1,
           1, 0, 0,
           1, 1, 1};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {6, 6, 1}, core::Float32, device);
        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(
                    im.Resize(0.5, t::geometry::Image::InterpType::Nearest),
                    std::runtime_error);
        } else {
            im = im.Resize(0.5, t::geometry::Image::InterpType::Nearest);
            EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                    output_ref, {3, 3, 1}, core::Float32, device)));
        }
    }
    {  // UInt8
        // clang-format off
        const std::vector<uint8_t> input_data =
          {0, 0, 128, 1, 1, 1,
           0, 1, 1, 0, 0, 1,
           128, 0, 0, 255, 0, 1,
           0, 1, 128, 0, 1, 128,
           1, 128, 1, 0, 255, 128,
           1, 1, 1, 1, 128, 1};
        const std::vector<uint8_t> output_ref_ipp =
          {0, 32, 1,
           32, 96, 32,
           33, 1, 128};
        const std::vector<uint8_t> output_ref_npp =
          {0, 33, 1,
           32, 96, 33,
           33, 1, 128};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {6, 6, 1}, core::UInt8, device);
        t::geometry::Image im(data);
        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.Resize(0.5, t::geometry::Image::InterpType::Super),
                         std::runtime_error);
        } else {
            t::geometry::Image im_low =
                    im.Resize(0.5, t::geometry::Image::InterpType::Super);
            utility::LogInfo("Super: {}",
                             im_low.AsTensor().View({3, 3}).ToString());

            if (device.GetType() == core::Device::DeviceType::CPU) {
                EXPECT_TRUE(im_low.AsTensor().AllClose(core::Tensor(
                        output_ref_ipp, {3, 3, 1}, core::UInt8, device)));
            } else {
                EXPECT_TRUE(im_low.AsTensor().AllClose(core::Tensor(
                        output_ref_npp, {3, 3, 1}, core::UInt8, device)));

                // Check output in the CI to see if other inteprolations works
                // with other platforms
                im_low = im.Resize(0.5, t::geometry::Image::InterpType::Linear);
                utility::LogInfo("Linear(impl. dependent): {}",
                                 im_low.AsTensor().View({3, 3}).ToString());

                im_low = im.Resize(0.5, t::geometry::Image::InterpType::Cubic);
                utility::LogInfo("Cubic(impl. dependent): {}",
                                 im_low.AsTensor().View({3, 3}).ToString());

                im_low =
                        im.Resize(0.5, t::geometry::Image::InterpType::Lanczos);
                utility::LogInfo("Lanczos(impl. dependent): {}",
                                 im_low.AsTensor().View({3, 3}).ToString());
            }
        }
    }
}

TEST_P(ImagePermuteDevices, PyrDown) {
    core::Device device = GetParam();

    {  // Float32
        // clang-format off
        const std::vector<float> input_data =
          {0, 0, 0, 1, 0, 1,
           0, 1, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 1,
           1, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 0, 1,
           1, 1, 1, 1, 1, 1};
        const std::vector<float> output_ref =
          {0.0596343, 0.244201, 0.483257,
           0.269109, 0.187536, 0.410317,
           0.752312, 0.347241, 0.521471};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {6, 6, 1}, core::Float32, device);
        t::geometry::Image im(data);

        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.PyrDown(), std::runtime_error);
        } else {
            im = im.PyrDown();
            EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                    output_ref, {3, 3, 1}, core::Float32, device)));
        }
    }

    {  // UInt8
        // clang-format off
        const std::vector<uint8_t> input_data =
          {0, 0, 0, 128, 0, 1,
           0, 128, 0, 0, 0, 1,
           0, 0, 0, 128, 0, 128,
           255, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 0, 1,
           1, 1, 255, 1, 128, 255};
        const std::vector<uint8_t> output_ref_ipp =
          {8, 31, 26,
           51, 25, 30,
           48, 38, 46};
        const std::vector<uint8_t> output_ref_npp =
          {7, 31, 25,
           51, 25, 29,
           48, 38, 46};
        // clang-format on

        core::Tensor data =
                core::Tensor(input_data, {6, 6, 1}, core::UInt8, device);
        t::geometry::Image im(data);

        if (!t::geometry::Image::HAVE_IPPICV &&
            device.GetType() ==
                    core::Device::DeviceType::CPU) {  // Not Implemented
            ASSERT_THROW(im.PyrDown(), std::runtime_error);
        } else {
            im = im.PyrDown();
            if (device.GetType() == core::Device::DeviceType::CPU) {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_ipp, {3, 3, 1}, core::UInt8, device)));
            } else {
                EXPECT_TRUE(im.AsTensor().AllClose(core::Tensor(
                        output_ref_npp, {3, 3, 1}, core::UInt8, device)));
            }
        }
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
            input_data, {rows, cols, channels}, core::Float32, device};
    t::geometry::Image input(t_input);
    t::geometry::Image output;

    // UInt8
    core::Tensor t_input_uint8_t =
            t_input.To(core::UInt8);  // normal static_cast is OK
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
            t_input.To(core::UInt16);  // normal static_cast is OK
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
TEST_P(ImagePermuteDevices, ToLegacy) {
    core::Device device = GetParam();
    // 2 byte dtype is general enough for uin8_t as well as float
    core::Dtype dtype = core::UInt16;

    // 2D tensor for 1 channel image
    core::Tensor t_1ch(std::vector<uint16_t>{0, 1, 2, 3, 4, 5}, {2, 3}, dtype,
                       device);

    // Test 1 channel image conversion
    t::geometry::Image im_1ch(t_1ch);
    geometry::Image leg_im_1ch = im_1ch.ToLegacy();
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
    geometry::Image leg_im_3ch = im_3ch.ToLegacy();
    for (int r = 0; r < im_3ch.GetRows(); ++r)
        for (int c = 0; c < im_3ch.GetCols(); ++c)
            for (int ch = 0; ch < im_3ch.GetChannels(); ++ch)
                EXPECT_EQ(im_3ch.At(r, c, ch).Item<uint16_t>(),
                          *leg_im_3ch.PointerAt<uint16_t>(c, r, ch));
}

TEST_P(ImagePermuteDevices, DepthToVertexNormalMaps) {
    core::Device device = GetParam();

    // clang-format off
    core::Tensor t_depth(std::vector<uint16_t>{
        0, 1, 2, 1, 0,
        0, 2, 4, 2, 0,
        0, 3, 6, 3, 29,
        0, 2, 4, 2, 0,
        0, 1, 2, 1, 0}, {5, 5, 1}, core::UInt16, device);
    core::Tensor t_depth_clipped_ref(std::vector<float>{
        0.0, 0.1, 0.2, 0.1, 0.0,
        0.0, 0.2, 0.4, 0.2, 0.0,
        0.0, 0.3, 0.6, 0.3, 0.0,
        0.0, 0.2, 0.4, 0.2, 0.0,
        0.0, 0.1, 0.2, 0.1, 0.0}, {5, 5, 1}, core::Float32, device);
    core::Tensor intrinsic(std::vector<double>{
            1.f, 0.f, 2.f,
            0.f, 1.f, 2.f,
            0.f, 0.f, 1.f}, {3, 3}, core::Float64, device);
    core::Tensor t_vertex_ref(std::vector<float>{
        0.0,0.0,0.0,  -0.1,-0.2,0.1,  0.0,-0.4,0.2,  0.1,-0.2,0.1,  0.0,0.0,0.0,
        0.0,0.0,0.0,  -0.2,-0.2,0.2,  0.0,-0.4,0.4,  0.2,-0.2,0.2,  0.0,0.0,0.0,
        0.0,0.0,0.0,  -0.3,0.0,0.3,   0.0,0.0,0.6,   0.3,0.0,0.3,   0.0,0.0,0.0,
        0.0,0.0,0.0,  -0.2,0.2,0.2,   0.0,0.4,0.4,   0.2,0.2,0.2,   0.0,0.0,0.0,
        0.0,0.0,0.0,  -0.1,0.2,0.1,   0.0,0.4,0.2,   0.1,0.2,0.1,   0.0,0.0,0.0
        }, {5, 5, 3}, core::Float32, device);
    core::Tensor t_normal_ref(std::vector<float>{
        0.0,0.0,0.0,  0.57735,0.57735,0.57735,      -0.894427,0.447214,0.0,         0.0,0.0,0.0,  0.0,0.0,0.0,
        0.0,0.0,0.0,  0.801784,0.534522,-0.267261,  -0.801784,0.267261,-0.534523,   0.0,0.0,0.0,  0.0,0.0,0.0,
        0.0,0.0,0.0,  0.57735,-0.57735,-0.57735,    -0.666667,-0.333333,-0.666667,  0.0,0.0,0.0,  0.0,0.0,0.0,
        0.0,0.0,0.0,  0.408248,-0.816497,0.408248,  -0.707107,-0.707107,-0.0,       0.0,0.0,0.0,  0.0,0.0,0.0,
        0.0,0.0,0.0,  0.0,0.0,0.0,                   0.0,0.0,0.0,                   0.0,0.0,0.0,  0.0,0.0,0.0
        }, {5, 5, 3}, core::Float32, device);
    // clang-format on
    t::geometry::Image depth{t_depth};

    float invalid_fill = 0.0f;
    auto depth_clipped = depth.ClipTransform(10.0, 0.0, 2.5, invalid_fill);

    EXPECT_TRUE(depth_clipped.AsTensor().AllClose(t_depth_clipped_ref));

    auto vertex_map = depth_clipped.CreateVertexMap(intrinsic, invalid_fill);
    EXPECT_TRUE(vertex_map.AsTensor().AllClose(t_vertex_ref));

    auto normal_map = vertex_map.CreateNormalMap(invalid_fill);
    EXPECT_TRUE(normal_map.AsTensor().AllClose(t_normal_ref));
}

TEST_P(ImagePermuteDevices, DISABLED_CreateVertexMap_Visual) {
    core::Device device = GetParam();

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image depth =
            t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0])
                    ->To(device);

    float invalid_fill = 0.0f;
    auto depth_clipped = depth.ClipTransform(1000.0, 0.0, 3.0, invalid_fill);

    core::Tensor intrinsic_t = CreateIntrinsics();
    auto vertex_map = depth_clipped.CreateVertexMap(intrinsic_t, invalid_fill);
    visualization::DrawGeometries(
            {std::make_shared<open3d::geometry::Image>(vertex_map.ToLegacy())});
}

TEST_P(ImagePermuteDevices, DISABLED_CreateNormalMap_Visual) {
    core::Device device = GetParam();

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image depth =
            t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0])
                    ->To(device);

    float invalid_fill = 0.0f;
    core::Tensor intrinsic_t = CreateIntrinsics();

    // We have to apply a bilateral filter, otherwise normals would be too
    // noisy.
    auto depth_clipped = depth.ClipTransform(1000.0, 0.0, 3.0, invalid_fill);
    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() == core::Device::DeviceType::CPU) {  // Not Implemented
        ASSERT_THROW(depth_clipped.FilterBilateral(5, 5.0, 10.0),
                     std::runtime_error);
    } else {
        auto depth_bilateral = depth_clipped.FilterBilateral(5, 5.0, 10.0);
        auto vertex_map_for_normal =
                depth_bilateral.CreateVertexMap(intrinsic_t, invalid_fill);
        auto normal_map = vertex_map_for_normal.CreateNormalMap(invalid_fill);

        // Use abs for better visualization
        normal_map.AsTensor() = normal_map.AsTensor().Abs();
        visualization::DrawGeometries(
                {std::make_shared<open3d::geometry::Image>(
                        normal_map.ToLegacy())});
    }
}

TEST_P(ImagePermuteDevices, DISABLED_ColorizeDepth) {
    core::Device device = GetParam();

    data::SampleRedwoodRGBDImages redwood_data;
    t::geometry::Image depth =
            t::io::CreateImageFromFile(redwood_data.GetDepthPaths()[0])
                    ->To(device);

    auto color_depth = depth.ColorizeDepth(1000.0, 0.0, 3.0);
    visualization::DrawGeometries({std::make_shared<open3d::geometry::Image>(
            color_depth.ToLegacy())});

    auto depth_clipped = depth.ClipTransform(1000.0, 0.0, 3.0, 0.0);
    auto color_depth_clipped = depth_clipped.ColorizeDepth(1.0, 0.0, 3.0);
    visualization::DrawGeometries({std::make_shared<open3d::geometry::Image>(
            color_depth_clipped.ToLegacy())});
}
}  // namespace tests
}  // namespace open3d
