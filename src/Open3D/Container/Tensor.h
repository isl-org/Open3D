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

namespace open3d {

/// A Tensor is a "view" of a data Blob with shape, stride, data_ptr.
/// Tensor can also be used to perform numerical operations.
class Tensor {
public:
    /// Constructor for creating a contiguous Tensor
    Tensor(const SizeVector& shape,
           const Dtype& dtype,
           const Device& device = Device("CPU:0"))
        : shape_(shape),
          strides_(DefaultStrides(shape)),
          dtype_(dtype),
          device_(device),
          blob_(std::make_shared<Blob>(
                  shape.NumElements() * DtypeUtil::ByteSize(dtype), device)) {
        data_ptr_ = blob_->v_;
    }

    /// Constructor for creating a contiguous Tensor with initial values
    template <typename T>
    Tensor(const std::vector<T>& init_vals,
           const SizeVector& shape,
           const Dtype& dtype,
           const Device& device = Device("CPU:0"))
        : Tensor(shape, dtype, device) {
        // Check number of elements
        if (init_vals.size() != shape_.NumElements()) {
            utility::LogError(
                    "Tensor initialization values' size {} does not match the "
                    "shape {}",
                    init_vals.size(), shape_.NumElements());
        }

        // Check data types
        AssertTemplateDtype<T>();

        // Copy data to blob
        MemoryManager::MemcpyFromHost(blob_->v_, device_, init_vals.data(),
                                      init_vals.size() * sizeof(T));
    }

    /// The fully specified constructor
    Tensor(const SizeVector& shape,
           const SizeVector& strides,
           void* data_ptr,
           const Dtype& dtype,
           const Device& device,
           const std::shared_ptr<Blob>& blob)
        : shape_(shape),
          strides_(strides),
          data_ptr_(data_ptr),
          dtype_(dtype),
          device_(device),
          blob_(blob) {
        if (!blob->IsPtrInBlob(data_ptr)) {
            utility::LogError("data_ptr not in the memory range of blob");
        }
    }

    /// Copy constructor with lvalue input, e.g. `Tensor dst(src)`
    Tensor(const Tensor& other);

    /// Copy constructor with rvalue input, e.g. `Tensor dst(src[0])`
    Tensor(Tensor&& other);

    /// Tensor assignment lvalue = lvalue, e.g. `tensor_a = tensor_b`
    Tensor& operator=(const Tensor& other) &;

    /// Tensor assignment lvalue = rvalue, e.g. `tensor_a = tensor_b[0]`
    Tensor& operator=(Tensor&& other) &;

    /// Tensor assignment rvalue = lvalue, e.g. `tensor_a[0] = tensor_b`
    Tensor& operator=(const Tensor& other) &&;

    /// Tensor assignment rvalue = rvalue, e.g. `tensor_a[0] = tensor_b[0]`
    Tensor& operator=(Tensor&& other) &&;

    /// Tensor assignment rvalue = rvalue_scalar, e.g. `tensor_a[0] = 100`
    /// Implicit casting is performed to the underlying dtype.
    ///
    /// Note that we don't have lvalue = rvalue_scalar, e.g. we don't support
    /// Tensor a_slice = tensor_a[0]; a_slice = 100;
    template <typename T>
    Tensor& operator=(const T& v) && {
        if (shape_.size() != 0) {
            utility::LogError(
                    "Assignment with scalar only works for scalar Tensor of "
                    "shape ()");
        }
        DISPATCH_DTYPE_TO_TEMPLATE(GetDtype(), [&]() {
            scalar_t casted_v = static_cast<scalar_t>(v);
            MemoryManager::MemcpyFromHost(GetDataPtr(), GetDevice(), &casted_v,
                                          sizeof(scalar_t));
        });
        return *this;
    }

    /// Assign (copy) values from another Tensor, shape, dtype, device may
    /// change. Slices of the original Tensor still keeps the original memory.
    /// After assignment, the Tensor will be contiguous.
    void Assign(const Tensor& other);

    /// \brief Fill the whole Tensor with a scalar value, the scalar will be
    /// casted to the Tensor's dtype.
    template <typename T>
    void Fill(const T& v) {
        DISPATCH_DTYPE_TO_TEMPLATE(GetDtype(), [&]() {
            scalar_t casted_v = static_cast<scalar_t>(v);
            Tensor tmp(std::vector<scalar_t>({casted_v}), SizeVector({}),
                       GetDtype(), GetDevice());
            *this = tmp;
        });
    }

    /// Broadcast Tensor to a new broadcastable shape
    Tensor Broadcast(const SizeVector& dst_shape) const;

    /// Copy Tensor to a specified device
    /// The resulting Tensor will be compacted and contiguous
    Tensor Copy(const Device& device) const;

    /// Copy Tensor values to current tensor for source tensor
    void CopyFrom(const Tensor& other);

    /// Clone Tensor to a specified device
    /// The resulting Tensor have the exact shape, stride and data_ptr to blob_
    /// beginning offset.
    Tensor Clone(const Device& device) const;

    std::string ToString(bool with_suffix = true,
                         const std::string& indent = "") const;

    /// Extract the i-th Tensor along the first axis, creating a new view
    Tensor operator[](int64_t i) const;

    /// Slice Tensor
    Tensor Slice(int64_t dim,
                 int64_t start,
                 int64_t stop,
                 int64_t step = 1) const;

    /// \brief Advanced indexing getter
    ///
    /// We use the Numpy advanced indexing symnatics, see:
    /// https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    ///
    /// Note: Only support 1D index tensors.
    /// Note: Only support advanced indices are all next to each other. E.g.
    /// ```
    /// A = np.ones((10, 20, 30, 40, 50))
    /// A[:, [1, 2], [2, 3], :, :]  # Supported,
    ///                               output_shape: [10, 2, 40, 50]
    ///                               slice_map:    [0, -1, 3, 4]
    /// A[:, [1, 2], :, [2, 3], :]  # No suport, output_shape: [2, 10, 30, 50]
    ///                             # For this case, a transpose op is necessary
    /// ```
    Tensor IndexGet(const std::vector<Tensor>& index_tensors) const;

    /// \brief Advanced indexing getter.
    ///
    /// We use the Numpy advanced indexing symnatics, see:
    /// https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    ///
    /// Note: Only support 1D index tensors.
    /// Note: Only support advanced indices are all next to each other.
    void IndexSet(const std::vector<Tensor>& index_tensors,
                  const Tensor& src_tensor);

    /// Helper function to return scalar value of a scalar Tensor, the Tensor
    /// mush have empty shape ()
    template <typename T>
    T Item() const {
        if (shape_.size() != 0) {
            utility::LogError("Item only works for scalar Tensor of shape ()");
        }
        AssertTemplateDtype<T>();
        T value;
        MemoryManager::MemcpyToHost(&value, data_ptr_, device_, sizeof(T));
        return value;
    }

    /// Retrive all values as an std::vector, for debugging and testing
    template <typename T>
    std::vector<T> ToFlatVector() const {
        AssertTemplateDtype<T>();
        std::vector<T> values(NumElements());
        MemoryManager::MemcpyToHost(
                values.data(), Contiguous().GetDataPtr(), GetDevice(),
                DtypeUtil::ByteSize(GetDtype()) * NumElements());
        return values;
    }

    /// Returns True if the underlying memory buffer is contiguous. A contiguous
    /// Tensor's data_ptr_ does not need to point to the beginning of blob_.
    bool IsContiguous() const { return DefaultStrides(shape_) == strides_; };

    /// Returns a contiguous Tensor containing the same data in the same device.
    /// If self tensor is already contiguous, the same underlying memory will be
    /// used.
    Tensor Contiguous() const;

    SizeVector GetShape() const { return shape_; }

    SizeVector GetStrides() const { return strides_; }

    void* GetDataPtr() { return data_ptr_; }

    const void* GetDataPtr() const { return data_ptr_; }

    Dtype GetDtype() const { return dtype_; }

    Device GetDevice() const { return device_; }

    std::shared_ptr<Blob> GetBlob() const { return blob_; }

    int64_t NumElements() const { return shape_.NumElements(); }

    int64_t NumDims() const { return shape_.size(); }

    template <typename T>
    void AssertTemplateDtype() const {
        if (DtypeUtil::FromType<T>() != dtype_) {
            utility::LogError(
                    "Requested values have type {} but Tensor has type {}",
                    DtypeUtil::ToString(DtypeUtil::FromType<T>()),
                    DtypeUtil::ToString(dtype_));
        }
        if (DtypeUtil::ByteSize(dtype_) != sizeof(T)) {
            utility::LogError("Internal error: element size mismatch {} != {}",
                              DtypeUtil::ByteSize(dtype_), sizeof(T));
        }
    }

    static SizeVector DefaultStrides(const SizeVector& shape);

protected:
    std::string ScalarPtrToString(const void* ptr) const;

protected:
    /// SizeVector of the Tensor. SizeVector[i] is the legnth of dimension i.
    SizeVector shape_;

    /// Stride of a Tensor.
    /// The stride of a n-dimensional tensor is also n-dimensional. Stride(i) is
    /// the number of elements (not bytes) to jump in a continuous memory space
    /// before eaching the next element in dimension i. For example, a 2x3x4
    /// float32 dense tensor has shape(2, 3, 4) and stride(12, 4, 1). A slicing
    /// operation performed on the tensor can change the shape and stride.
    SizeVector strides_;

    /// Data pointer points to the starting memory address in the blob
    void* data_ptr_;

    /// Data type
    Dtype dtype_;

    /// Device context (device type and id)
    /// TODO: This infomation is also available in blob_->Device, to remove?
    Device device_;

    /// Underlying memory buffer for Tensor.
    std::shared_ptr<Blob> blob_;
};

}  // namespace open3d
