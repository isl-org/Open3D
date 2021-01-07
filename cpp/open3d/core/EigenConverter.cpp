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

#include <type_traits>

#include "open3d/core/kernel/CPULauncher.h"

namespace open3d {
namespace core {
namespace eigen_converter {

template <typename T>
static std::vector<Eigen::Matrix<T, 3, 1>> TensorToEigenVector3xVector(
        const core::Tensor &tensor) {
    static_assert(std::is_same<T, double>::value || std::is_same<T, int>::value,
                  "Only supports double and int (Vector3d and Vector3i).");
    core::Dtype dtype;
    if (std::is_same<T, double>::value) {
        dtype = core::Dtype::Float64;
    } else if (std::is_same<T, int>::value) {
        dtype = core::Dtype::Int32;
    }
    if (dtype.ByteSize() * 3 != sizeof(Eigen::Matrix<T, 3, 1>)) {
        utility::LogError("Internal error: dtype size mismatch {} != {}.",
                          dtype.ByteSize() * 3, sizeof(Eigen::Matrix<T, 3, 1>));
    }
    tensor.AssertShapeCompatible({utility::nullopt, 3});

    // Eigen::Vector3x is not a "fixed-size vectorizable Eigen type" thus it is
    // safe to write directly into std vector memory, see:
    // https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html.
    std::vector<Eigen::Matrix<T, 3, 1>> eigen_vector(tensor.GetLength());
    core::Tensor t = tensor.Contiguous().To(dtype);
    MemoryManager::MemcpyToHost(eigen_vector.data(), t.GetDataPtr(),
                                t.GetDevice(),
                                t.GetDtype().ByteSize() * t.NumElements());
    return eigen_vector;
}

template <typename T>
static core::Tensor EigenVector3xVectorToTensor(
        const std::vector<Eigen::Matrix<T, 3, 1>> &values,
        core::Dtype dtype,
        const core::Device &device) {
    // Unlike TensorToEigenVector3xVector, more types can be supported here. To
    // keep consistency, we only allow double and int.
    static_assert(std::is_same<T, double>::value || std::is_same<T, int>::value,
                  "Only supports double and int (Vector3d and Vector3i).");
    // Init CPU Tensor.
    int64_t num_values = static_cast<int64_t>(values.size());
    core::Tensor tensor_cpu =
            core::Tensor::Empty({num_values, 3}, dtype, Device("CPU:0"));

    // Fill Tensor. This takes care of dtype conversion at the same time.
    core::Indexer indexer({tensor_cpu}, tensor_cpu,
                          core::DtypePolicy::ALL_SAME);
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        core::kernel::CPULauncher::LaunchIndexFillKernel(
                indexer, [&](void *ptr, int64_t workload_idx) {
                    // Fills the flattened tensor tensor_cpu[:] with dtype
                    // casting. tensor_cpu[:][i] corresponds to the (i/3)-th
                    // element's (i%3)-th coordinate value.
                    *static_cast<scalar_t *>(ptr) = static_cast<scalar_t>(
                            values[workload_idx / 3](workload_idx % 3));
                });
    });

    // Copy Tensor to device if necessary.
    return tensor_cpu.To(device);
}

std::vector<Eigen::Vector3d> TensorToEigenVector3dVector(
        const core::Tensor &tensor) {
    return TensorToEigenVector3xVector<double>(tensor);
}

std::vector<Eigen::Vector3i> TensorToEigenVector3iVector(
        const core::Tensor &tensor) {
    return TensorToEigenVector3xVector<int>(tensor);
}

core::Tensor EigenVector3dVectorToTensor(
        const std::vector<Eigen::Vector3d> &values,
        core::Dtype dtype,
        const core::Device &device) {
    return EigenVector3xVectorToTensor(values, dtype, device);
}

core::Tensor EigenVector3iVectorToTensor(
        const std::vector<Eigen::Vector3i> &values,
        core::Dtype dtype,
        const core::Device &device) {
    return EigenVector3xVectorToTensor(values, dtype, device);
}

}  // namespace eigen_converter
}  // namespace core
}  // namespace open3d
