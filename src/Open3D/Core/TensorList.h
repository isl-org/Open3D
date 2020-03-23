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

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/Broadcast.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Core/TensorKey.h"
namespace open3d {

/// A TensorList is an extendable tensor at the 0-th dimension.
/// It is similar to std::vector<Tensor>, but uses Open3D's tensor memory
/// management system.
///
/// Typical use cases:
/// - Pointcloud: (N, 3)
/// - Sparse Voxel Grid: (N, 8, 8, 8)
class TensorList {
public:
    /// Constructor for creating an (empty by default) tensor list.
    ///
    /// \param shape Shape for the contained tensors. e.g.
    /// (3) for a list points,
    /// (8, 8, 8) for a list of voxel blocks.
    /// \param dtype Type for the contained tensors. e.g. Dtype::Int64.
    /// \param device Device to store the contained tensors. e.g. "CPU:0".
    TensorList(const SizeVector& shape,
               Dtype dtype,
               const Device& device = Device("CPU:0"),
               const int64_t& size = 0);

    /// Constructor from a vector with broadcastable tensors.
    ///
    /// \param tensors A vector of tensors. The tensors must be broadcastable to
    /// a common shape, which will be set as the shape of the TensorList. The
    /// tensors must be on the same device and have the same dtype.
    /// \param device Device to store the contained tensors. e.g. "CPU:0".
    TensorList(const std::vector<Tensor>& tensors,
               const Device& device = Device("CPU:0"));

    /// Constructor from a list of broadcastable tensors.
    ///
    /// \param tensors A list of tensors. The tensors must be broadcastable to
    /// a common shape, which will be set as the shape of the TensorList.
    /// \param device Ddevice to store the contained tensors. e.g. "CPU:0".
    TensorList(const std::initializer_list<Tensor>& tensors,
               const Device& device = Device("CPU:0"));

    /// Constructor from iterators, an abstract wrapper for std vectors
    /// and initializer lists.
    template <class InputIterator>
    TensorList(InputIterator first,
               InputIterator last,
               const Device& device = Device("CPU:0"))
        : device_(device) {
        ConstructFromIterators(first, last);
    }

    /// Constructor from a raw internal tensor.
    /// The inverse of AsTensor().
    ///
    /// \param inplace:
    /// - If true (default), reuse the raw internal tensor. The input tensor
    /// must be contiguous.
    /// - If false, create a new contiguous internal tensor with precomputed
    /// reserved size.
    TensorList(const Tensor& internal_tensor, bool inplace = true);

    /// Factory constructor from a raw tensor
    static TensorList FromTensor(const Tensor& tensor, bool inplace = false);

    /// Copy constructor from a tensor list.
    /// Create a new tensor list with copy of data.
    TensorList(const TensorList& other);

    /// Deep copy
    void CopyFrom(const TensorList& other);

    /// TensorList assignment lvalue = lvalue, e.g.
    /// `tensorlist_a = tensorlist_b`,
    /// resulting in a shallow copy.
    /// We don't redirect Slice operation to tensors, so right value assignment
    /// is not explicitly supported.
    TensorList& operator=(const TensorList& other) &;

    /// Shallow copy
    void ShallowCopyFrom(const TensorList& other);

    /// Return the reference of the contained valid tensors with shared memory.
    Tensor AsTensor() const;

    /// Resize an existing tensor list.
    /// If the size increases, the increased part will be assigned 0.
    /// If the size decreases, the decreased part's value will be undefined.
    void Resize(int64_t n);

    /// Push back the copy of a tensor to the list.
    /// The tensor must broadcastable to the TensorList's shape.
    /// The tensor must be on the same device and have the same dtype.
    void PushBack(const Tensor& tensor);

    /// Concatenate two TensorLists.
    /// Return a new TensorList with data copied.
    /// Two TensorLists must have the same shape, type, and device.
    static TensorList Concatenate(const TensorList& a, const TensorList& b);

    /// Concatenate two TensorLists.
    TensorList operator+(const TensorList& other) const {
        return Concatenate(*this, other);
    }

    /// Extend the current TensorList with another TensorList appended to the
    /// end. The data is copied. The two TensorLists must have the same shape,
    /// dtype, and device.
    void Extend(const TensorList& other);

    TensorList& operator+=(const TensorList& other) {
        Extend(other);
        return *this;
    }

    /// Extract the i-th Tensor along the first axis, returning a new view.
    /// For advanced indexing like Slice, use tensorlist.AsTensor().Slice().
    Tensor operator[](int64_t index) const;

    /// Clear the tensor list by discarding all data and creating a empty one.
    void Clear();

    std::string ToString() const;

    SizeVector GetShape() const { return shape_; }

    Device GetDevice() const { return device_; }

    Dtype GetDtype() const { return dtype_; }

    int64_t GetSize() const { return size_; }

    int64_t GetReservedSize() const { return reserved_size_; }

    const Tensor& GetInternalTensor() const { return internal_tensor_; }

protected:
    // The shared internal constructor for iterators.
    template <class InputIterator>
    void ConstructFromIterators(InputIterator first, InputIterator last) {
        int64_t size = std::distance(first, last);
        if (size == 0) {
            utility::LogError(
                    "Empty input tensors cannot initialize a TensorList.");
        }

        // Infer size and reserved_size
        size_ = size;
        reserved_size_ = ReserveSize(size_);

        // Infer shape
        shape_ = std::accumulate(
                std::next(first), last, first->GetShape(),
                [](const SizeVector shape, const Tensor& tensor) {
                    return BroadcastedShape(std::move(shape),
                                            tensor.GetShape());
                });

        // Infer dtype
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

        // Construct internal tensor
        SizeVector expanded_shape = ExpandFrontDim(shape_, reserved_size_);
        internal_tensor_ = Tensor(expanded_shape, dtype_, device_);

        // Assign tensors
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

protected:
    /// The shape_ represents the shape for each element in the TensorList.
    /// The internal_tensor_'s shape is (reserved_size_, *shape_).
    SizeVector shape_;

    Dtype dtype_;
    Device device_;

    /// Maximum number of elements in TensorList.
    /// The internal_tensor_'s shape is (reserved_size_, *shape_).
    /// In general, reserved_size_ >= (1 << (ceil(log2(size_)) + 1))
    /// as conventionally done in std::vector.
    /// Examples: (size_, reserved_size_) = (3, 8), (4, 8), (5, 16).
    int64_t reserved_size_ = 0;

    /// Number of active (valid) elements in TensorList.
    /// The internal_tensor_ has shape (reserved_size_, *shape_), but only the
    /// front (size_, *shape_) is active.
    int64_t size_ = 0;

    /// The internal tensor for data storage.
    Tensor internal_tensor_;
};
}  // namespace open3d
