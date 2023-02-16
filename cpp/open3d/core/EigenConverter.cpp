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

#include <type_traits>

#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace core {
namespace eigen_converter {

/// Fills tensor[:][i] with func(i).
///
/// \param indexer The input tensor and output tensor to the indexer are the
/// same (as a hack), since the tensor are filled in-place.
/// \param func A function that takes pointer location and
/// workload index i, computes the value to fill, and fills the value at the
/// pointer location.
template <typename func_t>
static void LaunchIndexFillKernel(const Indexer &indexer, const func_t &func) {
    ParallelFor(Device("CPU:0"), indexer.NumWorkloads(),
                [&indexer, &func](int64_t i) {
                    func(indexer.GetInputPtr(0, i), i);
                });
}

template <typename T>
static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrix(const core::Tensor &tensor) {
    static_assert(std::is_same<T, double>::value ||
                          std::is_same<T, float>::value ||
                          std::is_same<T, int>::value,
                  "Only supports double, float and int (MatrixXd, MatrixXf and "
                  "MatrixXi).");
    core::Dtype dtype = core::Dtype::FromType<T>();

    core::SizeVector dim = tensor.GetShape();
    if (dim.size() != 2) {
        utility::LogError(
                " [TensorToEigenMatrix]: Number of dimensions supported = 2, "
                "but got {}.",
                dim.size());
    }

    core::Tensor tensor_cpu_contiguous =
            tensor.Contiguous().To(core::Device("CPU:0"), dtype);
    T *data_ptr = tensor_cpu_contiguous.GetDataPtr<T>();

    Eigen::Map<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            eigen_matrix(data_ptr, dim[0], dim[1]);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            eigen_matrix_copy(eigen_matrix);
    return eigen_matrix_copy;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrixXd(const core::Tensor &tensor) {
    return TensorToEigenMatrix<double>(tensor);
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrixXf(const core::Tensor &tensor) {
    return TensorToEigenMatrix<float>(tensor);
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrixXi(const core::Tensor &tensor) {
    return TensorToEigenMatrix<int>(tensor);
}

template <typename T, int N>
static std::vector<Eigen::Matrix<T, N, 1>> TensorToEigenVectorNxVector(
        const core::Tensor &tensor) {
    AssertTensorShape(tensor, {utility::nullopt, N});

    static_assert(
            (std::is_same<T, double>::value || std::is_same<T, int>::value) &&
                    N > 0,
            "Only supports double and int (VectorNd and VectorNi) with N>0.");
    core::Dtype dtype;
    if (std::is_same<T, double>::value) {
        dtype = core::Float64;
    } else if (std::is_same<T, int>::value) {
        dtype = core::Int32;
    }
    if (dtype.ByteSize() * N != sizeof(Eigen::Matrix<T, N, 1>)) {
        utility::LogError("Internal error: dtype size mismatch {} != {}.",
                          dtype.ByteSize() * N, sizeof(Eigen::Matrix<T, N, 1>));
    }

    // Eigen::VectorNx is not a "fixed-size vectorizable Eigen type" thus it is
    // safe to write directly into std vector memory, see:
    // https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html.
    std::vector<Eigen::Matrix<T, N, 1>> eigen_vector(tensor.GetLength());
    const core::Tensor t = tensor.To(dtype).Contiguous();
    MemoryManager::MemcpyToHost(eigen_vector.data(), t.GetDataPtr(),
                                t.GetDevice(),
                                t.GetDtype().ByteSize() * t.NumElements());
    return eigen_vector;
}

template <typename T, int N>
static core::Tensor EigenVectorNxVectorToTensor(
        const std::vector<Eigen::Matrix<T, N, 1>> &values,
        core::Dtype dtype,
        const core::Device &device) {
    // Unlike TensorToEigenVector3NVector, more types can be supported here. To
    // keep consistency, we only allow double and int.
    static_assert(
            (std::is_same<T, double>::value || std::is_same<T, int>::value) &&
                    N > 0,
            "Only supports double and int (VectorNd and VectorNi) with N>0.");
    // Init CPU Tensor.
    int64_t num_values = static_cast<int64_t>(values.size());
    core::Tensor tensor_cpu =
            core::Tensor::Empty({num_values, N}, dtype, Device("CPU:0"));

    // Fill Tensor. This takes care of dtype conversion at the same time.
    core::Indexer indexer({tensor_cpu}, tensor_cpu,
                          core::DtypePolicy::ALL_SAME);
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        LaunchIndexFillKernel(indexer, [&](void *ptr, int64_t workload_idx) {
            // Fills the flattened tensor tensor_cpu[:] with dtype
            // casting. tensor_cpu[:][i] corresponds to the (i/N)-th
            // element's (i%N)-th coordinate value.
            *static_cast<scalar_t *>(ptr) = static_cast<scalar_t>(
                    values[workload_idx / N](workload_idx % N));
        });
    });

    // Copy Tensor to device if necessary.
    return tensor_cpu.To(device);
}

std::vector<Eigen::Vector2d> TensorToEigenVector2dVector(
        const core::Tensor &tensor) {
    return TensorToEigenVectorNxVector<double, 2>(tensor);
}

std::vector<Eigen::Vector3d> TensorToEigenVector3dVector(
        const core::Tensor &tensor) {
    return TensorToEigenVectorNxVector<double, 3>(tensor);
}

std::vector<Eigen::Vector3i> TensorToEigenVector3iVector(
        const core::Tensor &tensor) {
    return TensorToEigenVectorNxVector<int, 3>(tensor);
}

std::vector<Eigen::Vector2i> TensorToEigenVector2iVector(
        const core::Tensor &tensor) {
    return TensorToEigenVectorNxVector<int, 2>(tensor);
}

core::Tensor EigenVector3dVectorToTensor(
        const std::vector<Eigen::Vector3d> &values,
        core::Dtype dtype,
        const core::Device &device) {
    return EigenVectorNxVectorToTensor(values, dtype, device);
}

core::Tensor EigenVector2dVectorToTensor(
        const std::vector<Eigen::Vector2d> &values,
        core::Dtype dtype,
        const core::Device &device) {
    return EigenVectorNxVectorToTensor(values, dtype, device);
}

core::Tensor EigenVector3iVectorToTensor(
        const std::vector<Eigen::Vector3i> &values,
        core::Dtype dtype,
        const core::Device &device) {
    return EigenVectorNxVectorToTensor(values, dtype, device);
}
core::Tensor EigenVector2iVectorToTensor(
        const std::vector<Eigen::Vector2i> &values,
        core::Dtype dtype,
        const core::Device &device) {
    return EigenVectorNxVectorToTensor(values, dtype, device);
}

}  // namespace eigen_converter
}  // namespace core
}  // namespace open3d
