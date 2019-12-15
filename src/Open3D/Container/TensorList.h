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

#include <cstddef>
#include <memory>
#include <string>

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

/// A TensorList is an extendable tensor in the 1st dimension.
/// It is similar to std::vector<Tensor>,
/// but uses open3d's tensor memory management system.
/// Typical use cases:
/// - Pointcloud: N x 1 x 3
/// - Sparse Voxel Grid: N x 8 x 8 x 8
class TensorList {
public:
    /// Constructor for creating an empty TensorList
    TensorList(const SizeVector& shape,
               const Dtype& dtype,
               const Device& device = Device("CPU:0"));

    /// Constructor from a vector with broadcastable tensors
    TensorList(std::vector<Tensor>& tensors,
               const Device& device = Device("CPU:0"));

    /// Directly construct from the copy of a raw internal tensor
    /// The inverse of AsTensor()
    TensorList(const Tensor& internal_tensor);

    /// Copy constructor from a TensorList
    /// Create a new TensorList with copy of data
    TensorList(const TensorList& other);

    /// Return the reference of the target TensorList, discarding original data
    /// The behavior is the same as '=' for python lists
    /// Note: this operation is the only one in TensorList that does not copy
    /// data
    TensorList& operator=(const TensorList& other);

    /// Return to reference of tensor with [0, size_-1]
    Tensor AsTensor() const;

    /// Resize an existing TensorList.
    void Resize(int64_t n);

    /// Push back a tensor to the list
    void PushBack(const Tensor& tensor);

    /// Concatenate two TensorLists
    /// Return a new TensorList with copy of data
    TensorList operator+(const TensorList& other) const;
    TensorList Concatenate(const TensorList& a, const TensorList& b);

    /// Concatenate two TensorLists and append the other to the end of *this
    void operator+=(const TensorList& other);

    /// Return the reference of one tensor
    Tensor operator[](int64_t index);

    /// Return a new TensorList with copy of data
    /// The behavior is the same as [Slice()] for python lists
    TensorList Slice(int64_t start, int64_t stop, int64_t step = 1);

    /// Return a new TensorList with copy of data
    TensorList Select(std::vector<int64_t>& indices) const;

    void Clear();

    const SizeVector& GetShape() const { return shape_; }
    const Device& GetDevice() const { return device_; }
    const Dtype& GetDtype() const { return dtype_; }
    int64_t GetSize() const { return size_; }
    int64_t GetReservedSize() const { return reserved_size_; }
    const Tensor& GetInternalTensor() const { return internal_tensor_; }

protected:
    void ExpandTensor(int64_t new_reserved_size) {
        SizeVector new_expanded_shape = expand_shape(shape_, new_reserved_size);
        Tensor new_internal_tensor =
                Tensor(new_expanded_shape, dtype_, device_);

        /// Copy data
        new_internal_tensor.Slice(0 /* dim */, 0, size_).AsRvalue() =
                internal_tensor_.Slice(0 /* dim */, 0, size_);
        internal_tensor_ = new_internal_tensor;
        reserved_size_ = new_reserved_size;
    }

    SizeVector expand_shape(const SizeVector& shape, int64_t new_dim_size = 0) {
        SizeVector expanded_shape = {new_dim_size};
        expanded_shape.insert(expanded_shape.end(), shape.begin(), shape.end());
        return expanded_shape;
    }

    int64_t reserve_size(int64_t n) {
        for (int i = 63; i >= 0; --i) {
            if (((1 << i) & n) > 0) {
                return 1 << (i + 1);
            }
        }
        return 0;
    }

    void check_index(int64_t index) const {
        if (index < -size_ || index > size_ - 1) {
            utility::LogError("Index {} out of bound ({}, {})", index, -size_,
                              size_ - 1);
        }
    }

protected:
    /// We always maintain an internal Tensor of (reserved_size_, shape_).
    /// However, we only allow accessing the Tensor of (size_, shape_).
    /// In general, we guarantee reserved_size_ = (1 << (ceil(log2(size_)) +
    /// 1)), as conventionally used in std::vector.
    /// Exampls: (size_, reserved_size_) = (3, 8), (4, 8), (5, 16).
    Tensor internal_tensor_;

    int64_t reserved_size_ = 0;
    int64_t size_ = 0;

    SizeVector shape_;
    Dtype dtype_;
    Device device_;
};
}  // namespace open3d
