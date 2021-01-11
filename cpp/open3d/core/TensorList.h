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

#include "open3d/core/Blob.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"

namespace open3d {
namespace core {

/// A tensorlist is a list of Tensors of the same shape, similar to
/// std::vector<Tensor>. Internally, a tensorlist stores the Tensors in one
/// bigger internal tensor, where the begin dimension of the internal tensor is
/// extendable.
///
/// Examples:
/// - A 3D point cloud with N points:
///   - element_shape        : (3,)
///   - reserved_size        : M, where M >= N
///   - internal_tensor.shape: (M, 3)
/// - Sparse voxel grid of N voxels:
///   - element_shape        : (8, 8, 8)
///   - reserved_size        : M, where M >= N
///   - internal_tensor.shape: (M, 8, 8, 8)
class TensorList {
public:
    /// Useful to support operator[] in a map.
    TensorList() : TensorList(SizeVector({}), Dtype::Float32) {}

    /// Constructs an empty tensorlist.
    ///
    /// \param element_shape Shape of the contained tensors, e.g. {3,}. 0-sized
    /// and scalar element_shape are allowed.
    /// \param dtype Data type of the contained tensors. e.g. Dtype::Float32.
    /// \param device Device of the contained tensors. e.g. Device("CPU:0").
    TensorList(const SizeVector& element_shape,
               Dtype dtype,
               const Device& device = Device("CPU:0"))
        : element_shape_(element_shape),
          size_(0),
          reserved_size_(ComputeReserveSize(0)),
          internal_tensor_(shape_util::Concat({reserved_size_}, element_shape_),
                           dtype,
                           device) {}

    /// Constructs a tensorlist from a vector of Tensors. The tensors must have
    /// the same shape, dtype and device. Values will be copied.
    ///
    /// \param tensors A vector of tensors. The tensors must have common shape,
    /// dtype and device.
    TensorList(const std::vector<Tensor>& tensors)
        : TensorList(tensors.begin(), tensors.end()) {}

    /// Constructs a tensorlist with specified size.
    ///
    /// \param size Size of the tensorlist.
    /// \param element_shape Shape of the contained tensors, e.g. {3,}. 0-sized
    /// and scalar element_shape are allowed.
    /// \param dtype Data type of the contained tensors. e.g. Dtype::Float32.
    /// \param device Device of the contained tensors. e.g. Device("CPU:0").
    TensorList(int64_t size,
               const SizeVector& element_shape,
               Dtype dtype,
               const Device& device = Device("CPU:0"))
        : element_shape_(element_shape),
          size_(size),
          reserved_size_(ComputeReserveSize(size)),
          internal_tensor_(shape_util::Concat({reserved_size_}, element_shape_),
                           dtype,
                           device) {}

    /// Constructs a tensorlist from a list of Tensors. The tensors must have
    /// the same shape, dtype and device. Values will be copied.
    ///
    /// \param tensors A list of tensors. The tensors must have common shape,
    /// dtype and device.
    TensorList(const std::initializer_list<Tensor>& tensors)
        : TensorList(tensors.begin(), tensors.end()) {}

    /// Constructs a tensorlist from Tensor iterator. The tensors must have
    /// the same shape, dtype and device. Values will be copied.
    ///
    /// \param begin Begin iterator.
    /// \param end End iterator.
    template <class InputIterator>
    TensorList(InputIterator begin, InputIterator end) {
        int64_t size = std::distance(begin, end);
        if (size == 0) {
            utility::LogError(
                    "Empty input tensors cannot initialize a tensorlist.");
        }

        // Set size_ and reserved_size_.
        size_ = size;
        reserved_size_ = ComputeReserveSize(size_);

        // Check shape consistency and set element_shape_.
        element_shape_ = begin->GetShape();
        std::for_each(begin, end, [&](const Tensor& tensor) -> void {
            if (tensor.GetShape() != element_shape_) {
                utility::LogError(
                        "Tensors must have the same shape {}, but got {}.",
                        element_shape_, tensor.GetShape());
            }
        });

        // Check dtype consistency.
        Dtype dtype = begin->GetDtype();
        std::for_each(begin, end, [&](const Tensor& tensor) -> void {
            if (tensor.GetDtype() != dtype) {
                utility::LogError(
                        "Tensors must have the same dtype {}, but got {}.",
                        dtype.ToString(), tensor.GetDtype().ToString());
            }
        });

        // Check device consistency.
        Device device = begin->GetDevice();
        std::for_each(begin, end, [&](const Tensor& tensor) -> void {
            if (tensor.GetDevice() != device) {
                utility::LogError(
                        "Tensors must have the same device {}, but got {}.",
                        device.ToString(), tensor.GetDevice().ToString());
            }
        });

        // Construct internal tensor.
        internal_tensor_ =
                Tensor(shape_util::Concat({reserved_size_}, element_shape_),
                       dtype, device);
        size_t i = 0;
        for (auto iter = begin; iter != end; ++iter, ++i) {
            internal_tensor_[i] = *iter;
        }
    }

    /// Factory function to create tensorlist from a Tensor.
    ///
    /// \param tensor The input tensor. The tensor must have at least one
    /// dimension (tensor.NumDims() >= 1). The first dimension of the tensor
    /// will be used as the "size" dimension of the tensorlist, while the
    /// remaining dimensions will be used as the element shape of the tensor
    /// list. For example, if the input tensor has shape (2, 3, 4), the
    /// resulting tensorlist will have size 2 and element shape (3, 4).
    ///
    /// \param inplace If `inplace == true`, the tensorlist shares the same
    /// memory with the input tensor. The input tensor must be contiguous. The
    /// resulting tensorlist cannot be extended. If `inplace == false`, the
    /// tensor values will be copied when creating the tensorlist.
    static TensorList FromTensor(const Tensor& tensor, bool inplace = false);

    /// Copy constructor for tensorlist. The internal tensor will share the same
    /// memory as the input. Also see: the copy constructor for Tensor.
    TensorList(const TensorList& other) = default;

    /// Move constructor for tensorlist. The internal tensor will share the same
    /// memory as the input. Also see: the move constructor for Tensor.
    TensorList(TensorList&& other) = default;

    /// Copy assignment operator. The internal tensor will share the same memory
    /// as the input.
    TensorList& operator=(const TensorList& other) & = default;

    /// Move assignment operator. The internal tensor will share the same memory
    /// as the input.
    TensorList& operator=(TensorList&& other) & = default;

    /// Performs actual copy from another tensorlist. The internal tensor will
    /// be explicitly copied. All attributes will be copied and replaced. The
    /// returned tensor will always be resizable.
    void CopyFrom(const TensorList& other);

    /// Duplicate the current tensorlist. Values will be copied. The returned
    /// tensor will always be resizable.
    TensorList Clone() const;

    /// Return the reference of the contained valid tensors with shared memory.
    Tensor AsTensor() const;

    /// Resize tensorlist.
    /// If the size increases, the increased part will be initialized with 0.
    /// If the size decreases, the reserved_size_ remain unchanged. This
    /// operation is only valid for resizable tensorlist.
    void Resize(int64_t new_size);

    /// Push back a tensor to the tensorlist. The values will be copied. This
    /// operation is only valid for resizable tensorlist.
    ///
    /// \param tensor The tensor to to be copied to the end of the tensorlist.
    /// The tensor must be of the same shape, dtype and device as the tensot
    /// list.
    void PushBack(const Tensor& tensor);

    /// Extend the current tensorlist with another tensorlist appended to the
    /// end. The data is copied. The two tensorlists must have the same
    /// element_shape, dtype, and device. This operation is only valid for
    /// resizable tensorlist.
    void Extend(const TensorList& other);

    /// Concatenate two tensorlists.
    /// Return a new tensorlists with data copied.
    /// Two tensorlists must have the same element_shape, type, and device.
    static TensorList Concatenate(const TensorList& a, const TensorList& b);

    /// Concatenate two tensorlists.
    TensorList operator+(const TensorList& other) const {
        return Concatenate(*this, other);
    }

    /// Inplace concatenate with another tensorlist. This operation is only
    /// valid for resizable tensorlist.
    TensorList& operator+=(const TensorList& other) {
        Extend(other);
        return *this;
    }

    /// Extract the i-th Tensor along the begin axis, returning a new view.
    /// For advanced indexing like Slice, use tensorlist.AsTensor().Slice().
    Tensor operator[](int64_t index) const;

    /// Clear the tensorlist by disgarding the internal tensor and resetting the
    /// size to 0. This operation is only valid for resizable tensorlist.
    void Clear();

    std::string ToString() const;

    SizeVector GetElementShape() const { return element_shape_; }

    void AssertElementShape(const SizeVector& expected_element_shape) const {
        if (expected_element_shape != element_shape_) {
            utility::LogError(
                    "TensorList has element shape {}, but is expected to have "
                    "element shape {}.",
                    element_shape_, expected_element_shape);
        }
    }

    void AssertDevice(const Device& expected_device) const {
        if (GetDevice() != expected_device) {
            utility::LogError(
                    "TensorList has device {}, but is expected to be {}.",
                    GetDevice().ToString(), expected_device.ToString());
        }
    }

    Device GetDevice() const { return internal_tensor_.GetDevice(); }

    Dtype GetDtype() const { return internal_tensor_.GetDtype(); }

    int64_t GetSize() const { return size_; }

    int64_t GetReservedSize() const { return reserved_size_; }

    const Tensor& GetInternalTensor() const { return internal_tensor_; }

    bool IsResizable() const { return is_resizable_; }

protected:
    /// Fully specified constructor.
    TensorList(const SizeVector element_shape,
               int64_t size,
               int64_t reserved_size,
               const Tensor& internal_tensor,
               bool is_resizable)
        : element_shape_(element_shape),
          size_(size),
          reserved_size_(reserved_size),
          internal_tensor_(internal_tensor),
          is_resizable_(is_resizable) {}

    /// Expand internal tensor to be larger or equal to the requested size. If
    /// the current reserved size is smaller than the requested size, the
    /// reserved size will be increased, a new internal tensor will be allocated
    /// and the original data will be copied. If the current reserved size is
    /// larger than or equal to the requested size, no operation will be
    /// performed.
    ///
    /// \param new_size The requested size.
    void ResizeWithExpand(int64_t new_size);

    /// Compute the reserved size for the desired number of tensors
    /// with reserved_size_ = (1 << (ceil(log2(size_)) + 1)).
    static int64_t ComputeReserveSize(int64_t size);

protected:
    /// The shape for each element tensor in the tensorlist.
    SizeVector element_shape_;

    /// Number of active (valid) elements in tensorlist.
    /// The internal_tensor_ has shape (reserved_size_, *shape_), but only the
    /// front (size_, *shape_) is active.
    int64_t size_ = 0;

    /// Maximum number of elements in tensorlist.
    ///
    /// The internal_tensor_'s shape is (reserved_size_, *element_shape_). In
    /// general, reserved_size_ >= (1 << (ceil(log2(size_)) + 1)) as
    /// conventionally done in std::vector.
    ///
    /// Examples: size_ = 3, reserved_size_ = 8
    ///           size_ = 4, reserved_size_ = 8
    ///           size_ = 5, reserved_size_ = 16
    int64_t reserved_size_ = 0;

    /// The internal tensor for data storage.
    Tensor internal_tensor_;

    /// Whether the tensorlist is resizable. Typically, if the tensorlist is
    /// created with pre-allocated shared buffer, the tensorlist is not
    /// resizable.
    bool is_resizable_ = true;
};
}  // namespace core
}  // namespace open3d
