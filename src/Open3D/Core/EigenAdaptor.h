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

#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Eigen.h"

namespace open3d {
template <typename Scalar>
Tensor FromEigen(const Eigen::Matrix<Scalar, -1, -1>& eigen_matrix) {
    auto dtype = DtypeUtil::FromType<Scalar>();

    // Eigen uses column major, so we fill in a transposed matrix
    // TODO: consider eigen allocator and alignment
    Tensor tensor = Tensor(
            SizeVector({eigen_matrix.cols(), eigen_matrix.rows()}), dtype);

    auto dst_ptr = tensor.GetBlob()->GetDataPtr();
    auto dst_stride = tensor.GetStrides()[0];

    auto src_ptr = reinterpret_cast<const void*>(eigen_matrix.data());
    auto src_stride = eigen_matrix.outerStride();

    for (int64_t i = 0; i < eigen_matrix.cols(); ++i) {
        MemoryManager::MemcpyFromHost(
                static_cast<Scalar*>(dst_ptr) + i * dst_stride,
                tensor.GetDevice(),
                static_cast<const Scalar*>(src_ptr) + i * src_stride,
                src_stride * sizeof(Scalar));
    }

    return tensor.T();
}

/// By default returns a double Matrix on CPU
template <typename Scalar>
Eigen::Matrix<Scalar, -1, -1> ToEigen(const Tensor& tensor) {
    auto dtype = DtypeUtil::FromType<Scalar>();
    if (dtype != tensor.GetDtype()) {
        utility::LogError("Eigen and tensor dtype mismatch.");
    }

    auto shape = tensor.GetShape();
    if (shape.size() != 2) {
        utility::LogError("A tensor must be 2D to be converted to a matrix.");
    }

    /// Copy from transposed tensor is more efficient for point clouds (N x 3)
    auto tensor_t = tensor.T().Contiguous();
    auto src_ptr = tensor_t.GetBlob()->GetDataPtr();
    auto src_stride = tensor_t.GetStrides()[0];

    Eigen::Matrix<Scalar, -1, -1> eigen_matrix(shape[0], shape[1]);
    auto dst_ptr = reinterpret_cast<void*>(eigen_matrix.data());
    auto dst_stride = eigen_matrix.outerStride();

    for (int64_t i = 0; i < eigen_matrix.cols(); ++i) {
        MemoryManager::MemcpyToHost(
                static_cast<Scalar*>(dst_ptr) + i * dst_stride,
                static_cast<const Scalar*>(src_ptr) + i * src_stride,
                tensor.GetDevice(), src_stride * sizeof(Scalar));
    }

    return eigen_matrix;
}
}  // namespace open3d
