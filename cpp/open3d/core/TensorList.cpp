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

#include "open3d/core/TensorList.h"

#include <string>

#include "open3d/core/SizeVector.h"

namespace open3d {
namespace core {

/// Asserts that the tensor list is resizable.
static void AssertIsResizable(const TensorList& tensorlist,
                              const std::string& func_name) {
    if (!tensorlist.IsResizable()) {
        utility::LogError(
                "TensorList::{}: TensorList is not resizable. Typically this "
                "tensorlist is created with shared memory from a Tensor.",
                func_name);
    }
}

TensorList TensorList::FromTensor(const Tensor& tensor, bool inplace) {
    SizeVector shape = tensor.GetShape();
    if (shape.size() == 0) {
        utility::LogError("Tensor should at least have one dimension.");
    }
    SizeVector element_shape =
            SizeVector(std::next(shape.begin()), shape.end());
    int64_t size = shape[0];

    if (inplace) {
        if (!tensor.IsContiguous()) {
            utility::LogError(
                    "Tensor must be contiguous for inplace tensorlist "
                    "construction.");
        }
        return TensorList(element_shape, size, size, tensor,
                          /*is_resizable=*/false);
    } else {
        int64_t reserved_size = TensorList::ComputeReserveSize(size);
        Tensor internal_tensor = Tensor::Empty(
                shape_util::Concat({reserved_size}, element_shape),
                tensor.GetDtype(), tensor.GetDevice());
        internal_tensor.Slice(0, 0, size) = tensor;
        return TensorList(element_shape, size, reserved_size, internal_tensor,
                          /*is_resizable=*/true);
    }
}

TensorList TensorList::Clone() const {
    TensorList copied(*this);
    copied.CopyFrom(*this);
    return copied;
}

void TensorList::CopyFrom(const TensorList& other) {
    *this = other;
    // Copy the full other.internal_tensor_, not just other.AsTensor().
    internal_tensor_ = other.internal_tensor_.Clone();
    // After copy, the resulting tensorlist is always resizable.
    is_resizable_ = true;
}

Tensor TensorList::AsTensor() const {
    return internal_tensor_.Slice(0, 0, size_);
}

void TensorList::Resize(int64_t new_size) {
    AssertIsResizable(*this, __FUNCTION__);

    // Increase internal tensor size.
    int64_t old_size = size_;
    ResizeWithExpand(new_size);
    internal_tensor_.Slice(0, old_size, new_size).Fill(0);
}

void TensorList::PushBack(const Tensor& tensor) {
    AssertIsResizable(*this, __FUNCTION__);

    if (element_shape_ != tensor.GetShape()) {
        utility::LogError(
                "TensorList has element shape {}, but tensor has shape {}.",
                element_shape_, tensor.GetShape());
    }
    if (GetDtype() != tensor.GetDtype()) {
        utility::LogError("TensorList has dtype {}, but tensor has shape {}.",
                          GetDtype().ToString(), tensor.GetDtype().ToString());
    }
    if (GetDevice() != tensor.GetDevice()) {
        utility::LogError("TensorList has device {}, but tensor has shape {}.",
                          GetDevice().ToString(),
                          tensor.GetDevice().ToString());
    }
    ResizeWithExpand(size_ + 1);
    internal_tensor_[size_ - 1] = tensor;  // same as operator[](-1) = tensor;
}

void TensorList::Extend(const TensorList& other) {
    AssertIsResizable(*this, __FUNCTION__);

    // Check consistency
    if (element_shape_ != other.GetElementShape()) {
        utility::LogError("TensorList shapes {} and {} are inconsistent.",
                          element_shape_, other.GetElementShape());
    }
    if (GetDevice() != other.GetDevice()) {
        utility::LogError("TensorList device {} and {} are inconsistent.",
                          GetDevice().ToString(), other.GetDevice().ToString());
    }
    if (GetDtype() != other.GetDtype()) {
        utility::LogError("TensorList dtype {} and {} are inconsistent.",
                          GetDtype().ToString(), other.GetDtype().ToString());
    }

    // Expand *this.
    int64_t other_size = other.GetSize();
    ResizeWithExpand(size_ + other_size);

    // Needs to slice other since *this and other can be the same tensorlist.
    // Assigning to a Tensor rvalue is an actual copy.
    internal_tensor_.Slice(0, size_ - other_size, size_) =
            other.AsTensor().Slice(0, 0, other_size);
}

TensorList TensorList::Concatenate(const TensorList& a, const TensorList& b) {
    // A full copy of a is required.
    TensorList result = a.Clone();
    result.Extend(b);
    return result;
}

Tensor TensorList::operator[](int64_t index) const {
    // WrapDim asserts index is within range.
    index = shape_util::WrapDim(index, size_);
    return internal_tensor_[index];
}

void TensorList::Clear() {
    AssertIsResizable(*this, __FUNCTION__);
    *this = TensorList(element_shape_, GetDtype(), GetDevice());
}

// Protected
void TensorList::ResizeWithExpand(int64_t new_size) {
    int64_t new_reserved_size = ComputeReserveSize(new_size);
    if (new_reserved_size <= reserved_size_) {
        size_ = new_size;
    } else {
        Tensor new_internal_tensor(
                shape_util::Concat({new_reserved_size}, element_shape_),
                GetDtype(), GetDevice());
        new_internal_tensor.Slice(0, 0, size_) =
                internal_tensor_.Slice(0, 0, size_);
        internal_tensor_ = new_internal_tensor;
        reserved_size_ = new_reserved_size;
        size_ = new_size;
    }
}

int64_t TensorList::ComputeReserveSize(int64_t n) {
    if (n < 0) {
        utility::LogError("Negative tensorlist size {} is not supported.", n);
    }

    int64_t base = 1;
    if (n > (base << 61)) {
        utility::LogError("Too large tensorlist size {} is not supported.", n);
    }

    for (int i = 63; i >= 0; --i) {
        // First nnz bit
        if (((base << i) & n) > 0) {
            if (n == (base << i)) {
                // Power of 2: 2 * n. For instance, 8 tensors will be
                // reserved for size=4
                return (base << (i + 1));
            } else {
                // Non-power of 2: ceil(log(2)) * 2. For instance, 16
                // tensors will be reserved for size=5
                return (base << (i + 2));
            }
        }
    }

    // No nnz bit: by default reserve 1 element.
    return 1;
}

std::string TensorList::ToString() const {
    return fmt::format(
            "TensorList[size: {}, element_shape: {}, dtype: {}, device: {}]",
            size_, element_shape_.ToString(), GetDtype().ToString(),
            GetDevice().ToString());
}

}  // namespace core
}  // namespace open3d
