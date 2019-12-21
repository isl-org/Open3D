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
#include "Open3D/Container/Broadcast.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"
namespace open3d {

/// A TensorList is an extendable tensor at the 0-th dimension.
/// It is similar to std::vector<Tensor>,
/// but uses Open3D's tensor memory management system.
/// Typical use cases:
/// - Pointcloud: N x 3
/// - Sparse Voxel Grid: N x 8 x 8 x 8
class TensorList {
public:
    /// Constructor for creating an empty tensor list.
    /// \param shape: shape for the contained tensors. e.g. (3) for a list of
    /// points, (8, 8, 8) for a list of voxel blocks.
    /// \param dtype: type for the contained tensors. e.g. Dtype::Int64.
    /// \param device: device to store the contained tensors. e.g. "CPU:0".
    TensorList(const SizeVector& shape,
               const Dtype& dtype,
               const Device& device = Device("CPU:0"));

    /// Constructor from a vector with broadcastable tensors.
    /// \param tensors: a vector of tensors with compatible shapes for
    /// broadcasting.
    /// \param device: device to store the contained tensors. e.g. "CPU:0".
    /// The tensors must have compatible shapes after broadcasting.
    /// The tensors must be on the same device and have the same dtype.
    TensorList(const std::vector<Tensor>& tensors,
               const Device& device = Device("CPU:0"));

    /// Constructor from a list of broadcastable tensors.
    /// \param tensors: a list of tensors contained in {},
    /// with compatible shapes for broadcasting.
    /// \param device: device to store the contained tensors. e.g. "CPU:0".
    /// The tensors must have compatible shapes after broadcasting.
    /// The tensors must be on the same device and have the same dtype.
    TensorList(const std::initializer_list<Tensor>& tensors,
               const Device& device = Device("CPU:0"));

    /// Constructor from iterators, an abstract wrapper for std vectors
    /// and initialize lists.
    /// The tensors pointed by iterators must have compatible shapes after
    /// broadcasting.
    /// The tensors must be on the same device and have the same dtype.
    template <class InputIterator>
    TensorList(InputIterator first,
               InputIterator last,
               const Device& device = Device("CPU:0"))
        : device_(device),
          /// Default empty tensor
          internal_tensor_(SizeVector(), Dtype::Int64, device) {
        ConstructFromIterators(first, last);
    }

    /// Directly construct from the copy of a raw internal tensor.
    /// The inverse of AsTensor().
    TensorList(const Tensor& internal_tensor);

    /// Copy constructor from a tensor list.
    /// Create a new tensor list with copy of data.
    TensorList(const TensorList& other);

    /// Return the reference of the target tensor list with shared memory,
    /// discarding original data.
    /// The behavior is the same as '=' for python lists.
    TensorList& operator=(const TensorList& other);

    /// Return the reference of the contained tensors with shared memory.
    Tensor AsTensor() const;

    /// Resize an existing tensor list.
    /// If the size increases, the increased part will be assigned 0.
    /// If the size decreases, the decreased part's value will be undefined.
    void Resize(int64_t n);

    /// Push back the copy of a tensor to the list.
    /// The tensor must have a compatible shape after broadcasting.
    /// The tensor must be on the same device and have the same dtype.
    void PushBack(const Tensor& tensor);

    /// Concatenate two tensor lists.
    /// Return a new tensor list with copy of data.
    /// Two tensor lists must have the same shape, type, and device.
    TensorList operator+(const TensorList& other) const;
    static TensorList Concatenate(const TensorList& a, const TensorList& b);

    /// Concatenate two tensor lists and append the copy of the other tensor
    /// list to the end of *this.
    /// Two TensorLists must have the same shape, dtype, and device.
    void operator+=(const TensorList& other);
    void Extend(const TensorList& b);

    /// Return the reference of one tensor with shared memory.
    Tensor operator[](int64_t index);

    /// Return a new tensor list with copy of data.
    /// The behavior is the same as [Slice()] for python lists.
    TensorList Slice(int64_t start, int64_t stop, int64_t step = 1);

    /// Return a new tensor list with copy of data.
    TensorList IndexGet(std::vector<int64_t>& indices) const;

    /// Clear the tensor list by discarding all data and creating a empty one.
    void Clear();

    const SizeVector& GetShape() const { return shape_; }
    const Device& GetDevice() const { return device_; }
    const Dtype& GetDtype() const { return dtype_; }
    int64_t GetSize() const { return size_; }
    int64_t GetReservedSize() const { return reserved_size_; }
    const Tensor& GetInternalTensor() const { return internal_tensor_; }

protected:
    /// The shared internal constructor for iterators.
    template <class InputIterator>
    void ConstructFromIterators(InputIterator first, InputIterator last) {
        int64_t size = std::distance(first, last);
        if (size == 0) {
            utility::LogError(
                    "Empty input tensors cannot initialize a TensorList.");
        }

        /// Infer size and reserved_size
        size_ = size;
        reserved_size_ = ReserveSize(size_);

        /// Infer shape
        shape_ = std::accumulate(
                std::next(first), last, first->GetShape(),
                [](const SizeVector shape, const Tensor& tensor) {
                    return BroadcastedShape(std::move(shape),
                                            tensor.GetShape());
                });

        /// Infer dtype
        dtype_ = first->GetDtype();
        bool dtype_consistent = std::accumulate(
                std::next(first), last, true,
                [&](bool same_type, const Tensor& tensor) {
                    return same_type && (dtype_ == tensor.GetDtype());
                });
        if (!dtype_consistent) {
            utility::LogError(
                    "Inconsistent tensor dtypes in tensors are not supported "
                    "in TensorList.");
        }

        /// Construct internal tensor
        SizeVector expanded_shape = ExpandFrontDim(shape_, reserved_size_);
        internal_tensor_ = Tensor(expanded_shape, dtype_, device_);

        /// Assign tensors
        size_t i = 0;
        for (auto iter = first; iter != last; ++iter, ++i) {
            internal_tensor_[i] = *iter;
        }
    }

    /// Expand the size of the internal tensor.
    void ExpandTensor(int64_t new_reserved_size);

    /// Expand the shape in the first indexing dimension.
    /// e.g. (8, 8, 8) -> (1, 8, 8, 8)
    static SizeVector ExpandFrontDim(const SizeVector& shape,
                                     int64_t new_dim_size = 1);

    /// Compute the reserved size for the desired number of tensors
    /// with reserved_size_ = (1 << (ceil(log2(size_)) + 1)).
    int64_t ReserveSize(int64_t n);

    /// Check if index is out of bound [0, size_).
    void CheckIndex(int64_t index) const;

protected:
    /// We always maintain an internal Tensor of (reserved_size_, **shape_).
    /// However, we only allow accessing the Tensor of (size_, **shape_).
    /// In general, reserved_size_ >= (1 << (ceil(log2(size_)) + 1))
    /// as conventionally done in std::vector.
    /// Examples: (size_, reserved_size_) = (3, 8), (4, 8), (5, 16).
    SizeVector shape_;
    Dtype dtype_;
    Device device_;

    int64_t size_ = 0;
    int64_t reserved_size_ = 0;

    Tensor internal_tensor_;
};
}  // namespace open3d
