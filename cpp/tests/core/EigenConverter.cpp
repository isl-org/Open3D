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

#include "open3d/core/EigenConverter.h"

#include <cmath>
#include <limits>

#include "open3d/core/Tensor.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class EigenConverterPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(EigenConverter,
                         EigenConverterPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(EigenConverterPermuteDevices, TensorToEigenMatrix) {
    core::Device device = GetParam();
    core::Device cpu_device = core::Device("CPU:0");

    // Device transfer and dtype conversions are handled.
    for (core::Dtype dtype : {core::Dtype::Float32, core::Dtype::Float64,
                              core::Dtype::Int32, core::Dtype::Int64}) {
        // TensorToEigenMatrix<T>(tensor).
        core::Tensor tensor = core::Tensor::Ones({5, 6}, dtype, device);
        Eigen::Matrix<double, 5, 6, 1> eigen =
                core::eigen_converter::TensorToEigenMatrix<double>(tensor);
        core::Tensor tensor_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen);
        // Memory is not shared.
        eigen(1, 1) = 0;
        EXPECT_TRUE(tensor_converted.AllClose(
                core::Tensor::Ones({5, 6}, core::Dtype::Float64, cpu_device)));

        // TensorToEigenMatrix4d.
        core::Tensor tensor4d = core::Tensor::Ones({4, 4}, dtype, device);
        Eigen::Matrix<double, 4, 4, 1> eigen4d =
                core::eigen_converter::TensorToEigenMatrix4d(tensor4d);
        core::Tensor tensor4d_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen4d);
        eigen4d(1, 1) = 0;
        EXPECT_TRUE(tensor4d_converted.AllClose(
                core::Tensor::Ones({4, 4}, core::Dtype::Float64, cpu_device)));

        // TensorToEigenMatrix4f.
        core::Tensor tensor4f = core::Tensor::Ones({4, 4}, dtype, device);
        Eigen::Matrix<float, 4, 4, 1> eigen4f =
                core::eigen_converter::TensorToEigenMatrix4f(tensor4f);
        core::Tensor tensor4f_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen4f);
        eigen4f(1, 1) = 0;
        EXPECT_TRUE(tensor4f_converted.AllClose(
                core::Tensor::Ones({4, 4}, core::Dtype::Float32, cpu_device)));

        // TensorToEigenMatrix4i.
        core::Tensor tensor4i = core::Tensor::Ones({4, 4}, dtype, device);
        Eigen::Matrix<int, 4, 4, 1> eigen4i =
                core::eigen_converter::TensorToEigenMatrix4i(tensor4i);
        core::Tensor tensor4i_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen4i);
        eigen4i(1, 1) = 0;
        EXPECT_TRUE(tensor4i_converted.AllClose(
                core::Tensor::Ones({4, 4}, core::Dtype::Int32, cpu_device)));

        // TensorToEigenMatrix6d.
        core::Tensor tensor6d = core::Tensor::Ones({6, 6}, dtype, device);
        Eigen::Matrix<double, 6, 6, 1> eigen6d =
                core::eigen_converter::TensorToEigenMatrix6d(tensor6d);
        core::Tensor tensor6d_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen6d);
        eigen6d(1, 1) = 0;
        EXPECT_TRUE(tensor6d_converted.AllClose(
                core::Tensor::Ones({6, 6}, core::Dtype::Float64, cpu_device)));

        // TensorToEigenMatrix6f.
        core::Tensor tensor6f = core::Tensor::Ones({6, 6}, dtype, device);
        Eigen::Matrix<float, 6, 6, 1> eigen6f =
                core::eigen_converter::TensorToEigenMatrix6f(tensor6f);
        core::Tensor tensor6f_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen6f);
        eigen6f(1, 1) = 0;
        EXPECT_TRUE(tensor6f_converted.AllClose(
                core::Tensor::Ones({6, 6}, core::Dtype::Float32, cpu_device)));

        // TensorToEigenMatrix6i.
        core::Tensor tensor6i = core::Tensor::Ones({6, 6}, dtype, device);
        Eigen::Matrix<int, 6, 6, 1> eigen6i =
                core::eigen_converter::TensorToEigenMatrix6i(tensor6i);
        core::Tensor tensor6i_converted =
                core::eigen_converter::EigenMatrixToTensor(eigen6i);
        eigen6i(1, 1) = 0;
        EXPECT_TRUE(tensor6i_converted.AllClose(
                core::Tensor::Ones({6, 6}, core::Dtype::Int32, cpu_device)));
    }
}

}  // namespace tests
}  // namespace open3d
