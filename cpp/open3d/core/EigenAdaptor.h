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
template <class T, int M, int N, int A>
Tensor FromEigen(const Eigen::Matrix<T, M, N, A>& matrix) {
    Dtype dtype = DtypeUtil::FromType<T>();
    Eigen::Matrix<T, M, N, Eigen::RowMajor> matrix_row_major = matrix;
    return Tensor(matrix_row_major.data(), {matrix.rows(), matrix.cols()},
                  dtype);
}

/// By default returns a double Matrix on CPU
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ToEigen(
        const Tensor& tensor, Eigen::StorageOptions align = Eigen::ColMajor) {
    Dtype dtype = DtypeUtil::FromType<T>();
    if (dtype != tensor.GetDtype()) {
        utility::LogError("Eigen and tensor dtype mismatch.");
    }
    SizeVector shape = tensor.GetShape();
    if (shape.size() != 2) {
        utility::LogError("A tensor must be 2D to be converted to a matrix.");
    }

    size_t num_bytes = DtypeUtil::ByteSize(dtype) * tensor.NumElements();
    Device device = tensor.GetDevice();
    if (align == Eigen::ColMajor) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
                matrix(shape[0], shape[1]);
        MemoryManager::MemcpyToHost(matrix.data(),
                                    tensor.T().Contiguous().GetDataPtr(),
                                    device, num_bytes);
        return matrix;
    } else if (align == Eigen::RowMajor) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                matrix(shape[0], shape[1]);
        MemoryManager::MemcpyToHost(matrix.data(),
                                    tensor.Contiguous().GetDataPtr(), device,
                                    num_bytes);
        return matrix;
    } else {
        utility::LogError("Only supports RowMajor or ColumnMajor.");
    }
}
}  // namespace open3d
