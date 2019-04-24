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
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/Utility/FileSystem.h"

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

    size_t num_elements = 1 << 24;

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

    size_t num_elements = 1 << 24;

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

    size_t num_elements = 1 << 24;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);

    vector<Eigen::Vector3d> normals(num_elements);
    Rand(normals, vmin, vmax, 1);

    geometry::PointCloud pc_cpu;
    pc_cpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::CPU);
    pc_cpu.normals_ = open3d::Normals(normals, open3d::cuda::DeviceID::CPU);
    pc_cpu.Transform(transformation);

    geometry::PointCloud pc_gpu;
    pc_gpu.points_ = open3d::Points(points, open3d::cuda::DeviceID::GPU_00);
    pc_gpu.normals_ = open3d::Normals(normals, open3d::cuda::DeviceID::GPU_00);
    pc_gpu.Transform(transformation);

    vector<Eigen::Vector3d> pc_gpu_points = pc_gpu.points_.Read();
    ExpectEQ(pc_cpu.points_.h_data, pc_gpu_points);

    vector<Eigen::Vector3d> pc_gpu_normals = pc_gpu.normals_.Read();
    ExpectEQ(pc_cpu.normals_.h_data, pc_gpu_normals);
}

// ----------------------------------------------------------------------------
// using the Blob<...>::Type
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, ComputePointCloudMeanAndCovariance) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    size_t num_elements = 1 << 24;

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

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, IO_CPU) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    const string file_name("test.ply");
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseError);

    geometry::PointCloud pc_write;
    geometry::PointCloud pc_read;

    size_t num_elements = 1 << 10;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);
    pc_write.points_ = open3d::Points(points, open3d::cuda::DeviceID::CPU);

    vector<Eigen::Vector3d> normals(num_elements);
    Rand(normals, vmin, vmax, 1);
    pc_write.normals_ = open3d::Normals(normals, open3d::cuda::DeviceID::CPU);

    // Read/WritePointCloud don't handle the colors at this time
    // vector<Eigen::Vector3d> colors(num_elements);
    // Rand(colors, vmin, vmax, 2);
    // pc_write.colors_ = open3d::Colors(colors, open3d::cuda::DeviceID::CPU);

    EXPECT_TRUE(io::WritePointCloud(file_name, pc_write));
    EXPECT_TRUE(io::ReadPointCloud(file_name, pc_read));
    EXPECT_TRUE(utility::filesystem::FileExists(file_name));

    ExpectEQ(points, pc_read.points_.h_data);
    ExpectEQ(normals, pc_read.normals_.h_data);
    // ExpectEQ(colors, pc_read.colors_.h_data);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
TEST(PointCloudCUDA, IO_GPU) {
    int nrGPUs = 0;
    cudaGetDeviceCount(&nrGPUs);
    EXPECT_TRUE(0 < nrGPUs);

    const string file_name("test.ply");
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseError);

    geometry::PointCloud pc_write;
    geometry::PointCloud pc_read;

    size_t num_elements = 1 << 10;

    Eigen::Vector3d vmin(-1.0, -1.0, -1.0);
    Eigen::Vector3d vmax(+1.0, +1.0, +1.0);

    vector<Eigen::Vector3d> points(num_elements);
    Rand(points, vmin, vmax, 0);
    pc_write.points_ = open3d::Points(points, open3d::cuda::DeviceID::GPU_00);

    vector<Eigen::Vector3d> normals(num_elements);
    Rand(normals, vmin, vmax, 1);
    pc_write.normals_ = open3d::Normals(normals, open3d::cuda::DeviceID::GPU_00);

    // Read/WritePointCloud don't handle the colors at this time
    // vector<Eigen::Vector3d> colors(num_elements);
    // Rand(colors, vmin, vmax, 2);
    // pc_write.colors_ = open3d::Colors(colors, open3d::cuda::DeviceID::GPU_00);

    EXPECT_TRUE(io::WritePointCloud(file_name, pc_write));
    EXPECT_TRUE(io::ReadPointCloud(file_name, pc_read));
    EXPECT_TRUE(utility::filesystem::FileExists(file_name));

    vector<Eigen::Vector3d> pc_read_points = pc_read.points_.Read();
    ExpectEQ(points, pc_read_points);

    vector<Eigen::Vector3d> pc_read_normals = pc_read.normals_.Read();
    ExpectEQ(normals, pc_read_normals);

    // vector<Eigen::Vector3d> pc_read_colors = pc_read.colors_.Read();
    // ExpectEQ(colors, pc_read_colors);
}

#endif
