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

#include "open3d/core/EigenConverter.h"

#include <cmath>
#include <limits>

#include "open3d/core/Tensor.h"
#include "tests/Tests.h"
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
    for (core::Dtype dtype :
         {core::Float32, core::Float64, core::Int32, core::Int64}) {
        for (const auto &shape :
             std::vector<core::SizeVector>({{0, 0}, {0, 1}, {1, 0}, {2, 3}})) {
            // TensorToEigenMatrixXd.
            core::Tensor tensor_d = core::Tensor::Ones(shape, dtype, device);
            auto eigen_d =
                    core::eigen_converter::TensorToEigenMatrixXd(tensor_d);
            core::Tensor tensor_converted_d =
                    core::eigen_converter::EigenMatrixToTensor(eigen_d);
            EXPECT_TRUE(tensor_converted_d.AllClose(
                    core::Tensor::Ones(shape, core::Float64, cpu_device)));

            // TensorToEigenMatrixXf.
            core::Tensor tensor_f = core::Tensor::Ones(shape, dtype, device);
            auto eigen_f =
                    core::eigen_converter::TensorToEigenMatrixXf(tensor_f);
            core::Tensor tensor_converted_f =
                    core::eigen_converter::EigenMatrixToTensor(eigen_f);
            EXPECT_TRUE(tensor_converted_f.AllClose(
                    core::Tensor::Ones(shape, core::Float32, cpu_device)));

            // TensorToEigenMatrixXi.
            core::Tensor tensor_i = core::Tensor::Ones(shape, dtype, device);
            auto eigen_i =
                    core::eigen_converter::TensorToEigenMatrixXi(tensor_i);
            core::Tensor tensor_converted_i =
                    core::eigen_converter::EigenMatrixToTensor(eigen_i);
            EXPECT_TRUE(tensor_converted_i.AllClose(
                    core::Tensor::Ones(shape, core::Int32, cpu_device)));
        }
    }

    core::Tensor tensor =
            core::Tensor::Full({5, 4}, 1.5, core::Float32, device);
    auto eigen = core::eigen_converter::TensorToEigenMatrixXi(tensor);
    core::Tensor tensor_converted =
            core::eigen_converter::EigenMatrixToTensor(eigen);
    EXPECT_TRUE(tensor_converted.AllClose(
            core::Tensor::Ones({5, 4}, core::Int32, cpu_device)));
}

TEST_P(EigenConverterPermuteDevices, EigenVectorToTensor) {
    // (3, 1) tensor.
    Eigen::Vector3i e_vector3i(0, 1, 2);
    core::Tensor t_vector3i =
            core::eigen_converter::EigenMatrixToTensor(e_vector3i);
    EXPECT_TRUE(
            t_vector3i.AllClose(core::Tensor::Init<int32_t>({{0}, {1}, {2}})));

    // (4, 1) tensor.
    Eigen::Vector4d e_vector4d(0.25, 1.00, 2.50, 3.75);
    core::Tensor t_vector4d =
            core::eigen_converter::EigenMatrixToTensor(e_vector4d);
    EXPECT_TRUE(t_vector4d.AllClose(
            core::Tensor::Init<double>({{0.25}, {1.00}, {2.50}, {3.75}})));
}

}  // namespace tests
}  // namespace open3d
