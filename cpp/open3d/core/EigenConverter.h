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

#pragma once

#include <Eigen/Core>
#include <vector>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorList.h"

namespace open3d {
namespace core {
namespace eigen_converter {

/// Converts a tensor of shape (3,) to Eigen::Vector3d. An exception will be
/// thrown if the tensor shape is not (3,).
Eigen::Vector3d TensorToEigenVector3d(const core::Tensor &tensor);

/// Converts a tensor of shape (N, 3) to std::vector<Eigen::Vector3d>. An
/// exception will be thrown if the tensor shape is not (N, 3). Regardless of
/// the tensor dtype, the output will be converted to to double.
///
/// \param tensor A tensor of shape (N, 3).
/// \return A vector of N Eigen::Vector3d values.
std::vector<Eigen::Vector3d> TensorToEigenVector3dVector(
        const core::Tensor &tensor);

/// Converts a vector of Eigen::Vector3d to a (N, 3) tensor. This
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

}  // namespace eigen_converter
}  // namespace core
}  // namespace open3d
