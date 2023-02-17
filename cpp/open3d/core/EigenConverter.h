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

#pragma once

#include <Eigen/Core>
#include <vector>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace eigen_converter {

/// \brief Converts a eigen matrix of shape (M, N) with alignment A and type T
/// to a tensor.
///
/// \param matrix An eigen matrix (e.g. Eigen::Matrix3f) or vector (e.g.
/// Eigen::Vector3d).
/// \return A tensor converted from the eigen matrix. The resuliting tensor is
/// always 2D.
template <class Derived>
core::Tensor EigenMatrixToTensor(const Eigen::MatrixBase<Derived> &matrix) {
    typedef typename Derived::Scalar Scalar;
    core::Dtype dtype = core::Dtype::FromType<Scalar>();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            matrix_row_major = matrix;
    return core::Tensor(matrix_row_major.data(), {matrix.rows(), matrix.cols()},
                        dtype);
}

/// \brief Converts a 2D tensor to Eigen::MatrixXd of same shape. Regardless of
/// the tensor dtype, the output will be converted to double.
///
/// \param tensor A 2D tensor.
/// \return Eigen::MatrixXd.
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrixXd(const core::Tensor &tensor);

/// \brief Converts a 2D tensor to Eigen::MatrixXf of same shape. Regardless of
/// the tensor dtype, the output will be converted to float.
///
/// \param tensor A 2D tensor.
/// \return Eigen::MatrixXf.
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrixXf(const core::Tensor &tensor);

/// \brief Converts a 2D tensor to Eigen::MatrixXi of same shape. Regardless of
/// the tensor dtype, the output will be converted to int.
///
/// \param tensor A 2D tensor.
/// \return Eigen::MatrixXi.
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrixXi(const core::Tensor &tensor);

/// \brief Converts a tensor of shape (N, 2) to std::vector<Eigen::Vector2d>. An
/// exception will be thrown if the tensor shape is not (N, 2). Regardless of
/// the tensor dtype, the output will be converted to double.
///
/// \param tensor A tensor of shape (N, 2).
/// \return A vector of N Eigen::Vector2d values.
std::vector<Eigen::Vector2d> TensorToEigenVector2dVector(
        const core::Tensor &tensor);

/// \brief Converts a tensor of shape (N, 3) to std::vector<Eigen::Vector3d>. An
/// exception will be thrown if the tensor shape is not (N, 3). Regardless of
/// the tensor dtype, the output will be converted to double.
///
/// \param tensor A tensor of shape (N, 3).
/// \return A vector of N Eigen::Vector3d values.
std::vector<Eigen::Vector3d> TensorToEigenVector3dVector(
        const core::Tensor &tensor);

/// \brief Converts a tensor of shape (N, 2) to std::vector<Eigen::Vector2i>. An
/// exception will be thrown if the tensor shape is not (N, 2). Regardless of
/// the tensor dtype, the output will be converted to int.
///
/// \param tensor A tensor of shape (N, 2).
/// \return A vector of N Eigen::Vector2i values.
std::vector<Eigen::Vector2i> TensorToEigenVector2iVector(
        const core::Tensor &tensor);

/// \brief Converts a tensor of shape (N, 3) to std::vector<Eigen::Vector3i>. An
/// exception will be thrown if the tensor shape is not (N, 3). Regardless of
/// the tensor dtype, the output will be converted to int.
///
/// \param tensor A tensor of shape (N, 3).
/// \return A vector of N Eigen::Vector3i values.
std::vector<Eigen::Vector3i> TensorToEigenVector3iVector(
        const core::Tensor &tensor);

/// \brief Converts a vector of Eigen::Vector3d to a (N, 3) tensor. This
/// function also takes care of dtype conversion and device transfer if
/// necessary.
///
/// \param values A vector of Eigen::Vector3d values, e.g. a list of 3D points.
/// \param dtype Dtype of the output tensor.
/// \param device Device of the output tensor.
/// \return A tensor of shape (N, 3) with the specified dtype and device.
core::Tensor EigenVector3dVectorToTensor(
        const std::vector<Eigen::Vector3d> &values,
        core::Dtype dtype,
        const core::Device &device);

/// \brief Converts a vector of Eigen::Vector2d to a (N, 2) tensor. This
/// function also takes care of dtype conversion and device transfer if
/// necessary.
///
/// \param values A vector of Eigen::Vector2d values, e.g. a list of UV
/// coordinates.
/// \param dtype Dtype of the output tensor.
/// \param device Device of the output tensor.
/// \return A tensor of shape (N, 2) with the specified dtype and device.
core::Tensor EigenVector2dVectorToTensor(
        const std::vector<Eigen::Vector2d> &values,
        core::Dtype dtype,
        const core::Device &device);

/// \brief Converts a vector of Eigen::Vector2i to a (N, 2) tensor. This
/// function also takes care of dtype conversion and device transfer if
/// necessary.
///
/// \param values A vector of Eigen::Vector2i values, e.g. a list of 2D points /
/// indices.
/// \param dtype Dtype of the output tensor.
/// \param device Device of the output tensor.
/// \return A tensor of shape (N, 2) with the specified dtype and device.
core::Tensor EigenVector2iVectorToTensor(
        const std::vector<Eigen::Vector2i> &values,
        core::Dtype dtype,
        const core::Device &device);

/// \brief Converts a vector of Eigen::Vector3i to a (N, 3) tensor. This
/// function also takes care of dtype conversion and device transfer if
/// necessary.
///
/// \param values A vector of Eigen::Vector3i values, e.g. a list of 3D points.
/// \param dtype Dtype of the output tensor.
/// \param device Device of the output tensor.
/// \return A tensor of shape (N, 3) with the specified dtype and device.
core::Tensor EigenVector3iVectorToTensor(
        const std::vector<Eigen::Vector3i> &values,
        core::Dtype dtype,
        const core::Device &device);

}  // namespace eigen_converter
}  // namespace core
}  // namespace open3d
