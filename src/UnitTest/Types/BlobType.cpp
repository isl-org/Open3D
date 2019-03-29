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
    open3d::Blob3d b3d;

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

    open3d::Blob3d b3d(num_elements, device_id);

    EXPECT_EQ(b3d.num_elements, num_elements);
    EXPECT_EQ(b3d.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b3d.h_data.size(), num_elements);
    EXPECT_FALSE(b3d.h_data.empty());
    EXPECT_TRUE(NULL == b3d.d_data);
    EXPECT_EQ(b3d.size(), num_elements);
}

// ----------------------------------------------------------------------------
// Initialization constructor - GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Initialization_constructor_GPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::GPU_00;

    open3d::Blob3d b3d(num_elements, device_id);

    EXPECT_EQ(b3d.num_elements, num_elements);
    EXPECT_EQ(b3d.device_id, open3d::cuda::DeviceID::GPU_00);
    EXPECT_EQ(b3d.h_data.size(), 0);
    EXPECT_TRUE(b3d.h_data.empty());
    EXPECT_TRUE(NULL != b3d.d_data);
    EXPECT_EQ(b3d.size(), num_elements);
}

// ----------------------------------------------------------------------------
// Initialization constructor - CPU and GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Initialization_constructor_CPU_GPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = (open3d::cuda::DeviceID::Type)(
            open3d::cuda::DeviceID::CPU | open3d::cuda::DeviceID::GPU_00);

    open3d::Blob3d b3d(num_elements, device_id);

    EXPECT_EQ(b3d.num_elements, num_elements);
    EXPECT_TRUE(open3d::cuda::DeviceID::CPU & b3d.device_id);
    EXPECT_TRUE(open3d::cuda::DeviceID::GPU_00 & b3d.device_id);
    EXPECT_EQ(b3d.h_data.size(), num_elements);
    EXPECT_FALSE(b3d.h_data.empty());
    EXPECT_TRUE(NULL != b3d.d_data);
    EXPECT_EQ(b3d.size(), num_elements);
}

// ----------------------------------------------------------------------------
// Copy constructor - CPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Copy_constructor_CPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::CPU;

    open3d::Blob3d b0(num_elements, device_id);
    Rand((double* const)b0.h_data.data(), b0.size() * 3, 0.0, 10.0, 0);

    open3d::Blob3d b1(b0);

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_EQ(b1.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b1.h_data.size(), num_elements);
    EXPECT_FALSE(b1.h_data.empty());
    EXPECT_TRUE(NULL == b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    ExpectEQ(b0.h_data, b1.h_data);
}

// ----------------------------------------------------------------------------
// Copy constructor - GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Copy_constructor_GPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::GPU_00;

    open3d::Blob3d b0(num_elements, device_id);

    // random initialization of b0
    vector<Eigen::Vector3d> b0_d_data(num_elements);
    Rand((double* const)b0_d_data.data(), num_doubles, 0.0, 10.0, 0);
    open3d::cuda::CopyHst2DevMemory((const double* const)b0_d_data.data(),
                                    b0.d_data, num_doubles);

    open3d::Blob3d b1(b0);

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_EQ(b1.device_id, open3d::cuda::DeviceID::GPU_00);
    EXPECT_EQ(b1.h_data.size(), 0);
    EXPECT_TRUE(b1.h_data.empty());
    EXPECT_TRUE(NULL != b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    vector<Eigen::Vector3d> b1_d_data(num_elements);
    open3d::cuda::CopyDev2HstMemory(b1.d_data, (double* const)b1_d_data.data(),
                                    num_doubles);

    ExpectEQ(b0_d_data, b1_d_data);
}

// ----------------------------------------------------------------------------
// Copy constructor - CPU and GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Copy_constructor_CPU_GPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = (open3d::cuda::DeviceID::Type)(
            open3d::cuda::DeviceID::CPU | open3d::cuda::DeviceID::GPU_00);

    open3d::Blob3d b0(num_elements, device_id);
    Rand((double* const)b0.h_data.data(), num_doubles, 0.0, 10.0, 0);
    open3d::cuda::CopyHst2DevMemory((const double* const)b0.h_data.data(),
                                    b0.d_data, num_doubles);

    open3d::Blob3d b1(b0);

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_TRUE(open3d::cuda::DeviceID::CPU & b1.device_id);
    EXPECT_TRUE(open3d::cuda::DeviceID::GPU_00 & b1.device_id);
    EXPECT_EQ(b1.h_data.size(), num_elements);
    EXPECT_FALSE(b1.h_data.empty());
    EXPECT_TRUE(NULL != b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    ExpectEQ(b0.h_data, b1.h_data);

    vector<Eigen::Vector3d> b1_d_data(num_elements);
    open3d::cuda::CopyDev2HstMemory(b1.d_data, (double* const)b1_d_data.data(),
                                    num_doubles);

    ExpectEQ(b0.h_data, b1_d_data);
}

// ----------------------------------------------------------------------------
// Reset - CPU and GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Reset) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = (open3d::cuda::DeviceID::Type)(
            open3d::cuda::DeviceID::CPU | open3d::cuda::DeviceID::GPU_00);

    open3d::Blob3d b0(num_elements, device_id);

    EXPECT_EQ(b0.num_elements, num_elements);
    EXPECT_TRUE(open3d::cuda::DeviceID::CPU & b0.device_id);
    EXPECT_TRUE(open3d::cuda::DeviceID::GPU_00 & b0.device_id);
    EXPECT_EQ(b0.h_data.size(), num_elements);
    EXPECT_FALSE(b0.h_data.empty());
    EXPECT_TRUE(NULL != b0.d_data);
    EXPECT_EQ(b0.size(), num_elements);

    b0.Reset();

    EXPECT_EQ(b0.num_elements, 0);
    EXPECT_EQ(b0.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b0.h_data.size(), 0);
    EXPECT_TRUE(b0.h_data.empty());
    EXPECT_TRUE(NULL == b0.d_data);
    EXPECT_EQ(b0.size(), 0);
}

// ----------------------------------------------------------------------------
// Assignment operator - CPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Assignment_operator_CPU) {
    size_t num_elements = 100;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::CPU;

    open3d::Blob3d b0(num_elements, device_id);
    Rand((double* const)b0.h_data.data(), b0.size() * 3, 0.0, 10.0, 0);

    open3d::Blob3d b1 = b0;

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_EQ(b1.device_id, open3d::cuda::DeviceID::CPU);
    EXPECT_EQ(b1.h_data.size(), num_elements);
    EXPECT_FALSE(b1.h_data.empty());
    EXPECT_TRUE(NULL == b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    ExpectEQ(b0.h_data, b1.h_data);
}

// ----------------------------------------------------------------------------
// Assignment operator - GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Assignment_operator_GPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::GPU_00;

    open3d::Blob3d b0(num_elements, device_id);

    // random initialization of b0
    vector<Eigen::Vector3d> b0_d_data(num_elements);
    Rand((double* const)b0_d_data.data(), num_doubles, 0.0, 10.0, 0);
    open3d::cuda::CopyHst2DevMemory((const double* const)b0_d_data.data(),
                                    b0.d_data, num_doubles);

    open3d::Blob3d b1 = b0;

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_EQ(b1.device_id, open3d::cuda::DeviceID::GPU_00);
    EXPECT_EQ(b1.h_data.size(), 0);
    EXPECT_TRUE(b1.h_data.empty());
    EXPECT_TRUE(NULL != b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    vector<Eigen::Vector3d> b1_d_data(num_elements);
    open3d::cuda::CopyDev2HstMemory(b1.d_data, (double* const)b1_d_data.data(),
                                    num_doubles);

    ExpectEQ(b0_d_data, b1_d_data);
}

// ----------------------------------------------------------------------------
// Assignment operator - CPU and GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Assignment_operator_CPU_GPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = (open3d::cuda::DeviceID::Type)(
            open3d::cuda::DeviceID::CPU | open3d::cuda::DeviceID::GPU_00);

    open3d::Blob3d b0(num_elements, device_id);
    Rand((double* const)b0.h_data.data(), num_doubles, 0.0, 10.0, 0);
    open3d::cuda::CopyHst2DevMemory((const double* const)b0.h_data.data(),
                                    b0.d_data, num_doubles);

    open3d::Blob3d b1 = b0;

    EXPECT_EQ(b1.num_elements, num_elements);
    EXPECT_TRUE(open3d::cuda::DeviceID::CPU & b1.device_id);
    EXPECT_TRUE(open3d::cuda::DeviceID::GPU_00 & b1.device_id);
    EXPECT_EQ(b1.h_data.size(), num_elements);
    EXPECT_FALSE(b1.h_data.empty());
    EXPECT_TRUE(NULL != b1.d_data);
    EXPECT_EQ(b1.size(), num_elements);

    ExpectEQ(b0.h_data, b1.h_data);

    vector<Eigen::Vector3d> b1_d_data(num_elements);
    open3d::cuda::CopyDev2HstMemory(b1.d_data, (double* const)b1_d_data.data(),
                                    num_doubles);

    ExpectEQ(b0.h_data, b1_d_data);
}

// ----------------------------------------------------------------------------
// Assignment operator - from vector - CPU only.
// ----------------------------------------------------------------------------
TEST(BlobType, Assignment_operator_vector_CPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::CPU;

    vector<Eigen::Vector3d> v(num_elements);
    Rand((double* const)v.data(), num_doubles, 0.0, 10.0, 0);

    open3d::Blob3d b0(num_elements, device_id);
    b0 = v;

    ExpectEQ(b0.h_data, v);
}

// ----------------------------------------------------------------------------
// Assignment operator - from vector - GPU only.
// ----------------------------------------------------------------------------
TEST(BlobType, Assignment_operator_vector_GPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = open3d::cuda::DeviceID::GPU_00;

    vector<Eigen::Vector3d> v(num_elements);
    Rand((double* const)v.data(), num_doubles, 0.0, 10.0, 0);

    open3d::Blob3d b0(num_elements, device_id);
    b0 = v;

    vector<Eigen::Vector3d> b0_d_data(num_elements);
    open3d::cuda::CopyDev2HstMemory(b0.d_data, (double* const)b0_d_data.data(),
                                    num_doubles);

    ExpectEQ(b0_d_data, v);
}

// ----------------------------------------------------------------------------
// Assignment operator - from vector - CPU and GPU.
// ----------------------------------------------------------------------------
TEST(BlobType, Assignment_operator_vector_CPU_GPU) {
    size_t num_elements = 100;
    size_t num_doubles = num_elements * 3;
    open3d::cuda::DeviceID::Type device_id = (open3d::cuda::DeviceID::Type)(
            open3d::cuda::DeviceID::CPU | open3d::cuda::DeviceID::GPU_00);

    vector<Eigen::Vector3d> v(num_elements);
    Rand((double* const)v.data(), num_doubles, 0.0, 10.0, 0);

    open3d::Blob3d b0(num_elements, device_id);
    b0 = v;

    vector<Eigen::Vector3d> b0_d_data(num_elements);
    open3d::cuda::CopyDev2HstMemory(b0.d_data, (double* const)b0_d_data.data(),
                                    num_doubles);

    ExpectEQ(b0.h_data, v);
    ExpectEQ(b0_d_data, v);
}
