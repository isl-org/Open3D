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

#include "Open3D/Core/TensorList.h"
#include "Open3D/Core/SizeVector.h"

namespace open3d {
// Public
TensorList::TensorList(const SizeVector& shape,
                       Dtype dtype,
                       const Device& device, /*= Device("CPU:0") */
                       const int64_t& size /* = 0 */)
    : shape_(shape),
      dtype_(dtype),
      device_(device),
      size_(size),
      reserved_size_(ReserveSize(size)) {
    internal_tensor_ =
            Tensor(ExpandFrontDim(shape_, reserved_size_), dtype_, device_);
}

TensorList::TensorList(const std::vector<Tensor>& tensors, const Device& device)
    : device_(device) {
    ConstructFromIterators(tensors.begin(), tensors.end());
}

TensorList::TensorList(const std::initializer_list<Tensor>& tensors,
                       const Device& device)
    : device_(device) {
    ConstructFromIterators(tensors.begin(), tensors.end());
}

TensorList::TensorList(const Tensor& internal_tensor, bool copy)
    : dtype_(internal_tensor.GetDtype()), device_(internal_tensor.GetDevice()) {
    SizeVector shape = internal_tensor.GetShape();

    size_ = shape[0];
    shape_ = SizeVector(std::next(shape.begin()), shape.end());

    if (copy) {
        // Construct the internal tensor with copy
        reserved_size_ = ReserveSize(size_);
        SizeVector expanded_shape = ExpandFrontDim(shape_, reserved_size_);
        internal_tensor_ = Tensor(expanded_shape, dtype_, device_);
        internal_tensor_.Slice(0 /* dim */, 0, size_) = internal_tensor;
    } else {
        // Directly reuse the slices
        if (!internal_tensor.IsContiguous()) {
            utility::LogError(
                    "Tensor must be contiguous for inplace TensorList "
                    "construction.");
        }
        reserved_size_ = size_;
        internal_tensor_ = internal_tensor;
    }
}

TensorList TensorList::FromTensor(const Tensor& tensor, bool inplace) {
    return TensorList(tensor, inplace);
}

TensorList::TensorList(const TensorList& other) { CopyFrom(other); }

void TensorList::CopyFrom(const TensorList& other) {
    shape_ = other.GetShape();
    dtype_ = other.GetDtype();
    device_ = other.GetDevice();
    size_ = other.GetSize();
    reserved_size_ = other.GetReservedSize();
    internal_tensor_.Assign(other.GetInternalTensor());
}

TensorList& TensorList::operator=(const TensorList& other) & {
    ShallowCopyFrom(other);
    return *this;
}

void TensorList::ShallowCopyFrom(const TensorList& other) {
    shape_ = other.GetShape();
    dtype_ = other.GetDtype();
    device_ = other.GetDevice();
    size_ = other.GetSize();
    reserved_size_ = other.GetReservedSize();
    internal_tensor_.ShallowCopyFrom(other.GetInternalTensor());
}

Tensor TensorList::AsTensor() const {
    return internal_tensor_.Slice(0 /* dim */, 0, size_);
}

void TensorList::Resize(int64_t n) {
    // Increase internal tensor size
    int64_t new_reserved_size = ReserveSize(n);
    if (new_reserved_size > reserved_size_) {
        ExpandTensor(new_reserved_size);
    }

    if (n > size_) {
        // Now new_reserved_size <= reserved_size, safe to fill in data
        internal_tensor_.Slice(0 /* dim */, size_, n).Fill(0);
    }
    size_ = n;
}

void TensorList::PushBack(const Tensor& tensor) {
    if (!CanBeBrocastedToShape(tensor.GetShape(), shape_)) {
        utility::LogError("Incompatible shape {} and {}", shape_,
                          tensor.GetShape());
    }

    int64_t new_reserved_size = ReserveSize(size_ + 1);
    if (new_reserved_size > reserved_size_) {
        ExpandTensor(new_reserved_size);
    }

    // Copy tensor
    internal_tensor_[size_] = tensor;
    ++size_;
}

TensorList TensorList::Concatenate(const TensorList& a, const TensorList& b) {
    TensorList result(a);
    result.Extend(b);
    return result;
}

void TensorList::Extend(const TensorList& other) {  // Check consistency
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

    // Ignore empty TensorList
    if (other.GetSize() == 0) {
        return;
    }

    // Shallow copy by default
    TensorList extension = other;

    // Make a deep copy to avoid corrupting duplicate data
    if (GetInternalTensor().GetDataPtr() ==
        other.GetInternalTensor().GetDataPtr()) {
        extension = TensorList(*this);
    }

    int64_t new_reserved_size = ReserveSize(size_ + extension.GetSize());
    if (new_reserved_size > reserved_size_) {
        ExpandTensor(new_reserved_size);
    }
    internal_tensor_.Slice(0 /* dim */, size_, size_ + extension.GetSize()) =
            extension.AsTensor();
    size_ = size_ + extension.GetSize();
}

Tensor TensorList::operator[](int64_t index) const {
    index = WrapDim(index, size_);  // WrapDim asserts index is within range.
    return internal_tensor_[index];
}

void TensorList::Clear() { *this = TensorList(shape_, dtype_, device_); }

// Protected
void TensorList::ExpandTensor(int64_t new_reserved_size) {
    if (new_reserved_size <= reserved_size_) {
        utility::LogError("New size {} is smaller than current size {}.",
                          new_reserved_size, reserved_size_);
    }
    SizeVector new_expanded_shape = ExpandFrontDim(shape_, new_reserved_size);
    Tensor new_internal_tensor = Tensor(new_expanded_shape, dtype_, device_);

    // Copy data
    new_internal_tensor.Slice(0 /* dim */, 0, size_) =
            internal_tensor_.Slice(0 /* dim */, 0, size_);
    internal_tensor_ = new_internal_tensor;
    reserved_size_ = new_reserved_size;
}

SizeVector TensorList::ExpandFrontDim(const SizeVector& shape,
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
    std::ostringstream rc;
    rc << fmt::format("\nTensorList[size={}, shape={}, {}, {}]", size_,
                      shape_.ToString(), DtypeUtil::ToString(dtype_),
                      GetDevice().ToString());
    return rc.str();
}
}  // namespace open3d
