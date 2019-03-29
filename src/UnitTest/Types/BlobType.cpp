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

#include "TestUtility/UnitTest.h"

#include "Open3D/Types/Blob.h"

#include <Eigen/Geometry>

#include <iostream>
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Make sure the types are PODs.
// ----------------------------------------------------------------------------
TEST(BlobType, is_POD) {
    EXPECT_FALSE(is_pod<open3d::Blob2i>());
    EXPECT_FALSE(is_pod<open3d::Blob3i>());
    EXPECT_FALSE(is_pod<open3d::Blob3d>());
}

// ----------------------------------------------------------------------------
// Default constructor.
// ----------------------------------------------------------------------------
TEST(BlobType, Default_constructor) {
    open3d::Blob<Eigen::Vector3d, double>::Type b3d;

    EXPECT_EQ(b3d.num_elements, 0);
    EXPECT_EQ(b3d.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b3d.h_data.size(), 0);
    EXPECT_TRUE(NULL == b3d.d_data);
    EXPECT_EQ(b3d.size(), 0);
}

// ----------------------------------------------------------------------------
// Initialization constructor - CPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Initialization_constructor_CPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::CPU;

    open3d::Blob<Eigen::Vector3d, double>::Type b3d(num_elements, device_id);

    EXPECT_EQ(b3d.num_elements, num_elements);
    EXPECT_EQ(b3d.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b3d.h_data.size(), num_elements);
    EXPECT_TRUE(NULL == b3d.d_data);
    EXPECT_EQ(b3d.size(), num_elements);
}

// ----------------------------------------------------------------------------
// Initialization constructor - GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Initialization_constructor_GPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::GPU_00;

    open3d::Blob<Eigen::Vector3d, double>::Type b3d(num_elements, device_id);

    EXPECT_EQ(b3d.num_elements, num_elements);
    EXPECT_EQ(b3d.device_id, open3d::cuda::DeviceID::GPU_00);
    EXPECT_EQ(b3d.h_data.size(), 0);
    EXPECT_TRUE(NULL != b3d.d_data);
    EXPECT_EQ(b3d.size(), num_elements);
}

// ----------------------------------------------------------------------------
// Copy constructor - CPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Copy_constructor_CPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::CPU;

    open3d::Blob<Eigen::Vector3d, double>::Type b0(num_elements, device_id);
    Rand((double* const)b0.h_data.data(), b0.size() * 3, 0.0, 10.0, 0);

    open3d::Blob<Eigen::Vector3d, double>::Type b1(b0);

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_EQ(b1.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b1.h_data.size(), num_elements);
    EXPECT_TRUE(NULL == b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    ExpectEQ(b0.h_data, b1.h_data);
}
