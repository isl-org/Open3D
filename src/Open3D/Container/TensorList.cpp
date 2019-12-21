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

#include "Open3D/Container/TensorList.h"

#include "Open3D/Container/Broadcast.h"

namespace open3d {
/// Public
TensorList::TensorList(const SizeVector& shape,
                       const Dtype& dtype,
                       const Device& device /*= Device("CPU:0") */)
    : shape_(shape),
      dtype_(dtype),
      device_(device),
      size_(0),
      reserved_size_(1),
      /// Default empty tensor
      internal_tensor_(SizeVector(), Dtype::Int64, device) {
    if (shape_.size() == 0) {
        utility::LogError(
                "Empty tensor shapes are not supported in TensorList.");
    }

    /// Construct internal tensor
    SizeVector expanded_shape = ExpandShape(shape_, reserved_size_);
    internal_tensor_ = Tensor(expanded_shape, dtype_, device_);
}

TensorList::TensorList(const std::vector<Tensor>& tensors, const Device& device)
    : device_(device),
      /// Default empty tensor
      internal_tensor_(SizeVector(), Dtype::Int64, device) {
    if (tensors.size() == 0) {
        utility::LogError(
                "Empty input tensors cannot initialize a TensorList.");
    }

    /// Infer size and reserved_size
    size_ = tensors.size();
    reserved_size_ = ReserveSize(size_);

    /// Infer shape
    shape_ = std::accumulate(
            std::next(tensors.begin()), tensors.end(), tensors[0].GetShape(),
            [](const SizeVector shape, const Tensor& tensor) {
                return BroadcastedShape(std::move(shape), tensor.GetShape());
            });

    if (shape_.size() == 0) {
        utility::LogError(
                "Empty input tensor shapes are not supported in TensorList.");
    }

    /// Infer dtype
    dtype_ = tensors[0].GetDtype();
    bool dtype_consistent = std::accumulate(
            std::next(tensors.begin()), tensors.end(), true,
            [&](bool same_type, const Tensor& tensor) {
                return same_type && (dtype_ == tensor.GetDtype());
            });
    if (!dtype_consistent) {
        utility::LogError(
                "Inconsistent tensor dtypes in tensors are not supported "
                "in TensorList.");
    }

    /// Construct internal tensor
    SizeVector expanded_shape = ExpandShape(shape_, reserved_size_);
    internal_tensor_ = Tensor(expanded_shape, dtype_, device_);

    /// Assign tensors
    for (size_t i = 0; i < size_; ++i) {
        internal_tensor_[i].AsRvalue() = tensors[i];
    }
}

TensorList::TensorList(const std::initializer_list<Tensor>& tensors,
                       const Device& device)
    : device_(device),
      /// Default empty tensor
      internal_tensor_(SizeVector(), Dtype::Int64, device) {
    if (tensors.size() == 0) {
        utility::LogError(
                "Empty input tensors cannot initialize a TensorList.");
    }

    /// Infer size and reserved_size
    size_ = tensors.size();
    reserved_size_ = ReserveSize(size_);

    /// Infer shape
    shape_ = std::accumulate(std::next(tensors.begin()), tensors.end(),
                             tensors.begin()->GetShape(),
                             [](const SizeVector shape, const Tensor& tensor) {
                                 return BroadcastedShape(std::move(shape),
                                                         tensor.GetShape());
                             });

    if (shape_.size() == 0) {
        utility::LogError(
                "Empty input tensor shapes are not supported in TensorList.");
    }

    /// Infer dtype
    dtype_ = tensors.begin()->GetDtype();
    bool dtype_consistent = std::accumulate(
            std::next(tensors.begin()), tensors.end(), true,
            [&](bool same_type, const Tensor& tensor) {
                return same_type && (dtype_ == tensor.GetDtype());
            });
    if (!dtype_consistent) {
        utility::LogError(
                "Inconsistent tensor dtypes in tensors are not supported "
                "in TensorList.");
    }

    /// Construct internal tensor
    SizeVector expanded_shape = ExpandShape(shape_, reserved_size_);
    internal_tensor_ = Tensor(expanded_shape, dtype_, device_);

    /// Assign tensors
    auto iter = tensors.begin();
    for (size_t i = 0; i < size_; ++i) {
        internal_tensor_[i].AsRvalue() = *iter;
        ++iter;
    }
}

TensorList::TensorList(const Tensor& tensor)
    : dtype_(tensor.GetDtype()),
      device_(tensor.GetDevice()),
      /// Default empty tensor
      internal_tensor_(SizeVector(), Dtype::Int64, tensor.GetDevice()) {
    SizeVector shape = tensor.GetShape();
    if (shape.size() <= 1) {
        utility::LogError(
                "Unable to construct TensorList from a Tensor with dim <= 1");
    }

    size_ = shape[0];
    reserved_size_ = ReserveSize(size_);
    shape_ = SizeVector(shape.begin() + 1, shape.end());

    /// Construct the internal tensor
    SizeVector expanded_shape = ExpandShape(shape_, reserved_size_);
    internal_tensor_ = Tensor(expanded_shape, dtype_, device_);
    internal_tensor_.Slice(0 /* dim */, 0, size_).AsRvalue() = tensor;
}

TensorList::TensorList(const TensorList& other)
    : shape_(other.GetShape()),
      dtype_(other.GetDtype()),
      device_(other.GetDevice()),
      size_(other.GetSize()),
      reserved_size_(other.GetReservedSize()),
      /// Default empty tensor
      internal_tensor_(SizeVector(), Dtype::Int64, other.GetDevice()) {
    internal_tensor_.Assign(other.GetInternalTensor());
}

TensorList& TensorList::operator=(const TensorList& other) {
    shape_ = other.GetShape();
    dtype_ = other.GetDtype();
    device_ = other.GetDevice();
    size_ = other.GetSize();
    reserved_size_ = other.GetReservedSize();
    internal_tensor_ = other.GetInternalTensor();
    return *this;
}

Tensor TensorList::AsTensor() const {
    return internal_tensor_.Slice(0 /* dim */, 0, size_);
}

void TensorList::Resize(int64_t n) {
    /// Increase internal tensor size
    int64_t new_reserved_size = ReserveSize(n);
    if (new_reserved_size > reserved_size_) {
        ExpandTensor(new_reserved_size);
    }

    if (n > size_) {
        /// Now new_reserved_size <= reserved_size, safe to fill in data
        internal_tensor_.Slice(0 /* dim */, size_, n).Fill(0);
    }
    size_ = n;
}

void TensorList::PushBack(const Tensor& tensor) {
    if (!IsCompatibleBroadcastShape(shape_, tensor.GetShape())) {
        utility::LogError("Incompatible shape {} and {}", shape_,
                          tensor.GetShape());
    }

    int64_t new_reserved_size = ReserveSize(size_ + 1);
    if (new_reserved_size > reserved_size_) {
        ExpandTensor(new_reserved_size);
    }

    /// Copy tensor
    internal_tensor_[size_].AsRvalue() = tensor;
    ++size_;
}

TensorList TensorList::operator+(const TensorList& other) const {
    /// Copy construct a new tensor list
    TensorList new_tensor_list(*this);
    new_tensor_list += other;
    return new_tensor_list;
}

TensorList TensorList::Concatenate(const TensorList& a, const TensorList& b) {
    return a + b;
}

void TensorList::operator+=(const TensorList& other) {
    /// Check consistency
    if (shape_ != other.GetShape()) {
        utility::LogError("TensorList shapes {} and {} are inconsistent.",
                          shape_, other.GetShape());
    }

    if (device_ != other.GetDevice()) {
        utility::LogError("TensorList device {} and {} are inconsistent.",
                          device_.ToString(), other.GetDevice().ToString());
    }

    if (dtype_ != other.GetDtype()) {
        utility::LogError("TensorList dtype {} and {} are inconsistent.",
                          DtypeUtil::ToString(dtype_),
                          DtypeUtil::ToString(other.GetDtype()));
    }

    /// Ignore empty TensorList
    if (other.GetSize() == 0) {
        return;
    }

    int64_t new_reserved_size = ReserveSize(size_ + other.GetSize());
    if (new_reserved_size > reserved_size_) {
        ExpandTensor(new_reserved_size);
    }
    internal_tensor_.Slice(0 /* dim */, size_, size_ + other.GetSize())
            .AsRvalue() = other.AsTensor();
    size_ = size_ + other.GetSize();
}

Tensor TensorList::operator[](int64_t index) {
    CheckIndex(index);
    if (index < 0) {
        index += size_;
    }
    return internal_tensor_[index];
}

TensorList TensorList::Slice(int64_t start,
                             int64_t stop,
                             int64_t step /* = 1 */) {
    CheckIndex(start);
    CheckIndex(stop);
    return TensorList(internal_tensor_.Slice(0 /* dim */, start, stop, step));
}

TensorList TensorList::IndexGet(std::vector<int64_t>& indices) const {
    std::vector<Tensor> tensors;
    for (auto& index : indices) {
        CheckIndex(index);
        if (index < 0) {
            index += size_;
        }
        tensors.push_back(internal_tensor_[index]);
    }

    return TensorList(tensors, device_);
}

void TensorList::Clear() { *this = TensorList(shape_, dtype_, device_); }

/// Protected
void TensorList::ExpandTensor(int64_t new_reserved_size) {
    if (new_reserved_size <= reserved_size_) {
        utility::LogError("New size {} is smaller than current size {}.",
                          new_reserved_size, reserved_size_);
    }
    SizeVector new_expanded_shape = ExpandShape(shape_, new_reserved_size);
    Tensor new_internal_tensor = Tensor(new_expanded_shape, dtype_, device_);

    /// Copy data
    new_internal_tensor.Slice(0 /* dim */, 0, size_).AsRvalue() =
            internal_tensor_.Slice(0 /* dim */, 0, size_);
    internal_tensor_ = new_internal_tensor;
    reserved_size_ = new_reserved_size;
}

SizeVector TensorList::ExpandShape(const SizeVector& shape,
                                   int64_t new_dim_size /* = 1 */) {
    SizeVector expanded_shape = {new_dim_size};
    expanded_shape.insert(expanded_shape.end(), shape.begin(), shape.end());
    return expanded_shape;
}

int64_t TensorList::ReserveSize(int64_t n) {
    if (n < 0) {
        utility::LogError("Negative tensor list size {} is unsupported.", n);
    }

    int64_t base = 1;
    if (n > (base << 61)) {
        utility::LogError("Too large tensor list size {} is unsupported.", n);
    }

    for (int i = 63; i >= 0; --i) {
        /// First nnz bit
        if (((base << i) & n) > 0) {
            if (n == (base << i)) {
                /// Power of 2: 2 * n. For instance, 8 tensors will be
                /// reserved for size=4
                return (base << (i + 1));
            } else {
                /// Non-power of 2: ceil(log(2)) * 2. For instance, 16
                /// tensors will be reserved for size=5
                return (base << (i + 2));
            }
        }
    }

    /// No nnz bit: by default reserve 1 element.
    return 1;
}

void TensorList::CheckIndex(int64_t index) const {
    if (index < -size_ || index > size_ - 1) {
        utility::LogError("Index {} out of bound ({}, {})", index, -size_,
                          size_ - 1);
    }
}

}  // namespace open3d
