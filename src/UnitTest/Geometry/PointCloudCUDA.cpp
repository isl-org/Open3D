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
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"

#include <algorithm>

using namespace open3d;
using namespace std;
using namespace unit_test;

#ifdef OPEN3D_USE_CUDA

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, GetMinBound) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    int num_elements = 1 << 24;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);

    geometry::PointCloud pc_cpu;
    pc_cpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::CPU);
    auto output_cpu = pc_cpu.GetMinBound();

    geometry::PointCloud pc_gpu;
    pc_gpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::GPU_00);
    auto output_gpu = pc_gpu.GetMinBound();

    ExpectEQ(output_cpu, output_gpu);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, GetMaxBound) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    int num_elements = 1 << 24;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);

    geometry::PointCloud pc_cpu;
    pc_cpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::CPU);
    auto output_cpu = pc_cpu.GetMaxBound();

    geometry::PointCloud pc_gpu;
    pc_gpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::GPU_00);
    auto output_gpu = pc_gpu.GetMaxBound();

    ExpectEQ(output_cpu, output_gpu);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, Transform) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    Eigen::Matrix4d transformation = Eigen::Matrix4d::Random();

    int num_elements = 10;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);

    geometry::PointCloud pc_cpu;
    pc_cpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::CPU);
    pc_cpu.Transform(transformation);

    geometry::PointCloud pc_gpu;
    pc_gpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::GPU_00);
    pc_gpu.Transform(transformation);

    EXPECT_TRUE(pc_cpu.points_ == pc_gpu.points_);
}

// ----------------------------------------------------------------------------
// using the Blob<...>::Type
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, ComputePointCloudMeanAndCovariance) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    int num_elements = 1 << 24;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);

    geometry::PointCloud pc_cpu;
    pc_cpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::CPU);
    auto output_cpu = geometry::ComputePointCloudMeanAndCovariance(pc_cpu);

    geometry::PointCloud pc_gpu;
    pc_gpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::GPU_00);
    auto output_gpu = geometry::ComputePointCloudMeanAndCovariance(pc_gpu);

    Eigen::Vector3d mean_cpu = get<0>(output_cpu);
    Eigen::Matrix3d covariance_cpu = get<1>(output_cpu);

    Eigen::Vector3d mean_gpu = get<0>(output_gpu);
    Eigen::Matrix3d covariance_gpu = get<1>(output_gpu);

    ExpectEQ(mean_cpu, mean_gpu);
    ExpectEQ(covariance_cpu, covariance_gpu);
}

// ----------------------------------------------------------------------------
// using the Tensor
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, ComputePointCloudMeanAndCovariance_Tensor) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    size_t num_elements = 1 << 24;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);

    open3d::Shape shape = {num_elements, 3};
    auto points_cpu = open3d::Tensor::create(shape, open3d::DataType::FP_64,
                                             open3d::cuda::DeviceID::CPU);

    geometry::PointCloud pc_cpu;
    pc_cpu.points_ = *static_pointer_cast<open3d::Points>(points_cpu);
    pc_cpu.points_ = points;
    auto output_cpu = geometry::ComputePointCloudMeanAndCovariance(pc_cpu);

    auto points_gpu = open3d::Tensor::create(shape, open3d::DataType::FP_64,
                                             open3d::cuda::DeviceID::GPU_00);

    geometry::PointCloud pc_gpu;
    pc_gpu.points_ = *static_pointer_cast<open3d::Points>(points_gpu);
    pc_gpu.points_ = points;
    auto output_gpu = geometry::ComputePointCloudMeanAndCovariance(pc_gpu);

    Eigen::Vector3d mean_cpu = get<0>(output_cpu);
    Eigen::Matrix3d covariance_cpu = get<1>(output_cpu);

    Eigen::Vector3d mean_gpu = get<0>(output_gpu);
    Eigen::Matrix3d covariance_gpu = get<1>(output_gpu);

    ExpectEQ(mean_cpu, mean_gpu);
    ExpectEQ(covariance_cpu, covariance_gpu);
}

#endif
