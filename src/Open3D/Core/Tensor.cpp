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

#include "Open3D/Core/Tensor.h"

#include <sstream>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/Blob.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Kernel/Kernel.h"
#include "Open3D/Core/ShapeUtil.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/TensorKey.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

/// Tensor assignment lvalue = lvalue, e.g. `tensor_a = tensor_b`
Tensor& Tensor::operator=(const Tensor& other) & {
    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    blob_ = other.blob_;
    data_ptr_ = other.data_ptr_;
    return *this;
}

/// Tensor assignment lvalue = rvalue, e.g. `tensor_a = tensor_b[0]`
Tensor& Tensor::operator=(Tensor&& other) & {
    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    blob_ = other.blob_;
    data_ptr_ = other.data_ptr_;
    return *this;
}

/// Tensor assignment rvalue = lvalue, e.g. `tensor_a[0] = tensor_b`
Tensor& Tensor::operator=(const Tensor& other) && {
    kernel::Copy(other, *this);
    return *this;
}

/// Tensor assignment rvalue = rvalue, e.g. `tensor_a[0] = tensor_b[0]`
Tensor& Tensor::operator=(Tensor&& other) && {
    kernel::Copy(other, *this);
    return *this;
}

Tensor Tensor::GetItem(const TensorKey& tk) const {
    if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
        return IndexExtract(0, tk.GetIndex());
    } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
        TensorKey tk_new = tk.UpdateWithDimSize(shape_[0]);
        return Slice(0, tk_new.GetStart(), tk_new.GetStop(), tk_new.GetStep());
    } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
        Tensor index_tensor(*tk.GetIndexTensor());
        return IndexGet({index_tensor});
    } else {
        utility::LogError("Internal error: wrong TensorKeyMode.");
    }
}

Tensor Tensor::GetItem(const std::vector<TensorKey>& tks) const {
    if (std::any_of(tks.begin(), tks.end(), [](const TensorKey& tk) {
            return tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor;
        })) {
        // If tks contains one or more IndexTensor, the advanced indexing mode
        // is enabled. Under Advanced indexing mode, we do some preprocessing
        // with regular slicing, before sending to the advanced indexing engine.
        //
        // 1) TensorKey::Index: convert to a TensorKey::IndexTensor with the
        //    specified index.
        // 2) TensorKey::Slice: if the slice is non-full slice, slice the tensor
        //    first and then use full slice for the advanced indexing engine.
        //
        // e.g.
        // dst = src[1,     0:2,   [1, 2]]
        //           ^      ^      ^
        //           Index  Slice  IndexTensor
        // is done in two steps:
        // temp = src[:, 0:2, :]
        // dst = temp[[1], :, [1, 2]]

        std::vector<TensorKey> preprocess_tks;

        // Performs `temp = src[:, 0:2, :]`, see the example above.
        for (const TensorKey& tk : tks) {
            if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
                preprocess_tks.push_back(TensorKey::Slice(None, None, None));
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
                preprocess_tks.push_back(tk);
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
                preprocess_tks.push_back(TensorKey::Slice(None, None, None));
            } else {
                utility::LogError("Internal error: wrong TensorKeyMode.");
            }
        }
        Tensor preprocess_t = GetItem(preprocess_tks);

        // Performs `dst = temp[[1], :, [1, 2]]`, see the example above.
        std::vector<Tensor> index_tensors;
        for (const TensorKey& tk : tks) {
            if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
                index_tensors.push_back(
                        Tensor(std::vector<int64_t>({tk.GetIndex()}), {1},
                               Dtype::Int64, GetDevice()));
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
                index_tensors.push_back(Tensor(std::vector<int64_t>{},
                                               Dtype::Int64, GetDevice()));
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
                index_tensors.push_back(Tensor(*tk.GetIndexTensor()));
            } else {
                utility::LogError("Internal error: wrong TensorKeyMode.");
            }
        }

        // Calls Advanced indexing engine.
        return preprocess_t.IndexGet(index_tensors);
    }

    Tensor t = *this;
    int64_t slice_dim = 0;
    for (const TensorKey& tk : tks) {
        if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
            t = t.IndexExtract(slice_dim, tk.GetIndex());
        } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
            TensorKey tk_new = tk.UpdateWithDimSize(t.shape_[slice_dim]);
            t = t.Slice(slice_dim, tk_new.GetStart(), tk_new.GetStop(),
                        tk_new.GetStep());
            slice_dim++;
        } else {
            utility::LogError("Internal error: wrong TensorKeyMode.");
        }
    }
    return t;
}

Tensor Tensor::SetItem(const Tensor& value) {
    this->AsRvalue() = value;
    return *this;
}

Tensor Tensor::SetItem(const TensorKey& tk, const Tensor& value) {
    if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
        Tensor index_tensor(*tk.GetIndexTensor());
        IndexSet({index_tensor}, value);
    } else {
        this->GetItem(tk) = value;
    }
    return *this;
}

Tensor Tensor::SetItem(const std::vector<TensorKey>& tks, const Tensor& value) {
    if (std::any_of(tks.begin(), tks.end(), [](const TensorKey& tk) {
            return tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor;
        })) {
        // Advanced indexing mode, see Tensor::GetItem for detailed docs.
        std::vector<TensorKey> preprocess_tks;

        for (const TensorKey& tk : tks) {
            if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
                preprocess_tks.push_back(TensorKey::Slice(None, None, None));
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
                preprocess_tks.push_back(tk);
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
                preprocess_tks.push_back(TensorKey::Slice(None, None, None));
            } else {
                utility::LogError("Internal error: wrong TensorKeyMode.");
            }
        }
        Tensor preprocess_t = GetItem(preprocess_tks);

        std::vector<Tensor> index_tensors;
        for (const TensorKey& tk : tks) {
            if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
                index_tensors.push_back(
                        Tensor(std::vector<int64_t>({tk.GetIndex()}), {1},
                               Dtype::Int64, GetDevice()));
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
                index_tensors.push_back(Tensor(std::vector<int64_t>{},
                                               Dtype::Int64, GetDevice()));
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
                index_tensors.push_back(Tensor(*tk.GetIndexTensor()));
            } else {
                utility::LogError("Internal error: wrong TensorKeyMode.");
            }
        }

        preprocess_t.IndexSet(index_tensors, value);
    } else {
        this->GetItem(tks) = value;
    }

    return *this;
}

/// Assign (copy) values from another Tensor, shape, dtype, device may change.
void Tensor::Assign(const Tensor& other) {
    shape_ = other.shape_;
    strides_ = DefaultStrides(shape_);
    dtype_ = other.dtype_;
    blob_ = std::make_shared<Blob>(
            shape_.NumElements() * DtypeUtil::ByteSize(dtype_),
            other.GetDevice());
    data_ptr_ = blob_->GetDataPtr();
    kernel::Copy(other, *this);
}

/// Broadcast Tensor to a new broadcastable shape
Tensor Tensor::Broadcast(const SizeVector& dst_shape) const {
    if (!shape_util::CanBeBrocastedToShape(shape_, dst_shape)) {
        utility::LogError("Cannot broadcast shape {} to shape {}.",
                          shape_.ToString(), dst_shape);
    }
    Tensor dst_tensor(dst_shape, dtype_, GetDevice());
    dst_tensor.AsRvalue() = *this;
    return dst_tensor;
}

Tensor Tensor::Expand(const SizeVector& dst_shape) const {
    if (!shape_util::CanBeBrocastedToShape(shape_, dst_shape)) {
        utility::LogError("Cannot expand shape {} to shape {}.",
                          shape_.ToString(), dst_shape);
    }
    int64_t src_ndims = NumDims();
    int64_t dst_ndims = dst_shape.size();
    int64_t omitted_ndims = dst_ndims - src_ndims;

    // Fill 1 in shape for omitted dimensions in front.
    // Noe that unexpanded_new_shape is not the expanded shape. The expanded
    // shape is the dst_shape.
    SizeVector unexpanded_new_shape(dst_ndims, 1);
    for (int64_t i = 0; i < src_ndims; ++i) {
        unexpanded_new_shape[i + omitted_ndims] = shape_[i];
    }

    // Fill 0 in strides for omitted dimensions in front.
    SizeVector new_strides(dst_ndims, 0);
    for (int64_t i = 0; i < src_ndims; ++i) {
        new_strides[i + omitted_ndims] = strides_[i];
    }

    // Set stride to 0 if the dimension is expanded.
    for (int64_t i = 0; i < dst_ndims; ++i) {
        if (unexpanded_new_shape[i] == 1 && dst_shape[i] != 1) {
            new_strides[i] = 0;
        }
    }

    return AsStrided(dst_shape, new_strides);
}

Tensor Tensor::Reshape(const SizeVector& dst_shape) const {
    SizeVector inferred_dst_shape =
            shape_util::InferShape(dst_shape, NumElements());
    bool can_restride;
    SizeVector new_strides;
    std::tie(can_restride, new_strides) =
            ComputeNewStrides(shape_, strides_, inferred_dst_shape);
    if (can_restride) {
        return AsStrided(inferred_dst_shape, new_strides);
    } else {
        return Contiguous().View(inferred_dst_shape);
    }
}

Tensor Tensor::View(const SizeVector& dst_shape) const {
    SizeVector inferred_dst_shape =
            shape_util::InferShape(dst_shape, NumElements());
    bool can_restride;
    SizeVector new_strides;
    std::tie(can_restride, new_strides) =
            ComputeNewStrides(shape_, strides_, inferred_dst_shape);
    if (can_restride) {
        return AsStrided(inferred_dst_shape, new_strides);
    } else {
        utility::LogError(
                "View shape {} is not compatible with Tensor's size {} and "
                "sride {}, at least one dimension spacs across two contiguous "
                "subspaces. Use Reshape() instead.",
                dst_shape, shape_, strides_);
    }
}

Tensor Tensor::Copy(const Device& device) const {
    Tensor dst_tensor(shape_, dtype_, device);
    kernel::Copy(*this, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::To(Dtype dtype, bool copy) const {
    if (!copy && dtype_ == dtype) {
        return *this;
    }
    Tensor dst_tensor(shape_, dtype, GetDevice());
    kernel::Copy(*this, dst_tensor);
    return dst_tensor;
}

void Tensor::CopyFrom(const Tensor& other) { AsRvalue() = other; }

void Tensor::ShallowCopyFrom(const Tensor& other) {
    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    blob_ = other.blob_;
    data_ptr_ = other.data_ptr_;
}

Tensor Tensor::Contiguous() const {
    if (IsContiguous()) {
        // Returns a shallow copy of the current Tensor
        return Tensor(shape_, strides_, data_ptr_, dtype_, blob_);
    } else {
        // Compact the tensor to contiguous on the same device
        return Copy(GetDevice());
    }
}

SizeVector Tensor::DefaultStrides(const SizeVector& shape) {
    SizeVector strides(shape.size());
    int64_t stride_size = 1;
    for (int64_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride_size;
        // Handles 0-sized dimensions
        stride_size *= std::max<int64_t>(shape[i - 1], 1);
    }
    return strides;
}

std::pair<bool, SizeVector> Tensor::ComputeNewStrides(
        const SizeVector& old_shape,
        const SizeVector& old_strides,
        const SizeVector& new_shape) {
    if (old_shape.empty()) {
        return std::make_pair(true, SizeVector(new_shape.size(), 1));
    }

    // NOTE: Stride is arbitrary in the numel() == 0 case. To match NumPy
    // behavior we copy the strides if the size matches, otherwise we use the
    // stride as if it were computed via resize. This could perhaps be combined
    // with the below code, but the complexity didn't seem worth it.
    int64_t numel = old_shape.NumElements();
    if (numel == 0 && old_shape == new_shape) {
        return std::make_pair(true, old_strides);
    }

    SizeVector new_strides(new_shape.size());
    if (numel == 0) {
        for (int64_t view_d = new_shape.size() - 1; view_d >= 0; view_d--) {
            if (view_d == (int64_t)(new_shape.size() - 1)) {
                new_strides[view_d] = 1;
            } else {
                new_strides[view_d] =
                        std::max<int64_t>(new_shape[view_d + 1], 1) *
                        new_strides[view_d + 1];
            }
        }
        return std::make_pair(true, new_strides);
    }

    int64_t view_d = new_shape.size() - 1;
    // Stride for each subspace in the chunk
    int64_t chunk_base_stride = old_strides.back();
    // Numel in current chunk
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;
    for (int64_t tensor_d = old_shape.size() - 1; tensor_d >= 0; tensor_d--) {
        tensor_numel *= old_shape[tensor_d];
        // If end of tensor size chunk, check view
        if ((tensor_d == 0) ||
            (old_shape[tensor_d - 1] != 1 &&
             old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
            while (view_d >= 0 &&
                   (view_numel < tensor_numel || new_shape[view_d] == 1)) {
                new_strides[view_d] = view_numel * chunk_base_stride;
                view_numel *= new_shape[view_d];
                view_d--;
            }
            if (view_numel != tensor_numel) {
                return std::make_pair(false, SizeVector());
            }
            if (tensor_d > 0) {
                chunk_base_stride = old_strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if (view_d != -1) {
        return std::make_pair(false, SizeVector());
    }
    return std::make_pair(true, new_strides);
}

std::string Tensor::ToString(bool with_suffix,
                             const std::string& indent) const {
    std::ostringstream rc;
    if (GetDevice().GetType() == Device::DeviceType::CUDA || !IsContiguous()) {
        Tensor host_contiguous_tensor = Copy(Device("CPU:0"));
        rc << host_contiguous_tensor.ToString(false, "");
    } else {
        if (shape_.NumElements() == 0) {
            rc << indent;
            rc << "0-element Tensor";
        } else if (shape_.size() == 0) {
            rc << indent;
            rc << ScalarPtrToString(data_ptr_);
        } else if (shape_.size() == 1) {
            const char* ptr = static_cast<const char*>(data_ptr_);
            rc << "[";
            std::string delim = "";
            int64_t element_byte_size = DtypeUtil::ByteSize(dtype_);
            for (int64_t i = 0; i < shape_.NumElements(); ++i) {
                rc << delim << ScalarPtrToString(ptr);
                delim = " ";
                ptr += element_byte_size;
            }
            rc << "]";
        } else {
            rc << "[";
            std::string delim = "";
            std::string child_indent = "";
            for (int64_t i = 0; i < shape_[0]; ++i) {
                rc << delim << child_indent
                   << this->operator[](i).ToString(false, indent + " ");
                delim = ",\n";
                child_indent = indent + " ";
            }
            rc << "]";
        }
    }
    if (with_suffix) {
        rc << fmt::format("\nTensor[shape={}, stride={}, {}, {}, {}]",
                          shape_.ToString(), strides_.ToString(),
                          DtypeUtil::ToString(dtype_), GetDevice().ToString(),
                          data_ptr_);
    }
    return rc.str();
}

std::string Tensor::ScalarPtrToString(const void* ptr) const {
    std::string str = "";
    if (dtype_ == Dtype::Bool) {
        str = *static_cast<const unsigned char*>(ptr) ? "True" : "False";
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE(dtype_, [&]() {
            str = fmt::format("{}", *static_cast<const scalar_t*>(ptr));
        });
    }
    return str;
}

Tensor Tensor::operator[](int64_t i) const { return IndexExtract(0, i); }

Tensor Tensor::IndexExtract(int64_t dim, int64_t idx) const {
    if (shape_.size() == 0) {
        utility::LogError("Tensor has shape (), cannot be indexed.");
    }
    dim = WrapDim(dim, NumDims());
    idx = WrapDim(idx, shape_[dim]);

    SizeVector new_shape(shape_);
    new_shape.erase(new_shape.begin() + dim);
    SizeVector new_strides(strides_);
    new_strides.erase(new_strides.begin() + dim);
    void* new_data_ptr = static_cast<char*>(data_ptr_) +
                         strides_[dim] * DtypeUtil::ByteSize(dtype_) * idx;
    return Tensor(new_shape, new_strides, new_data_ptr, dtype_, blob_);
}

Tensor Tensor::Slice(int64_t dim,
                     int64_t start,
                     int64_t stop,
                     int64_t step) const {
    if (shape_.size() == 0) {
        utility::LogError("Slice cannot be applied to 0-dim Tensor");
    }
    dim = WrapDim(dim, NumDims());
    if (dim < 0 || dim >= static_cast<int64_t>(shape_.size())) {
        utility::LogError("Dim {} is out of bound for SizeVector of length {}",
                          dim, shape_.size());
    }
    // TODO: support negative step sizes
    if (step == 0) {
        utility::LogError("Step size cannot be 0");
    }
    start = WrapDim(start, shape_[dim]);
    stop = WrapDim(stop, shape_[dim], /*inclusive=*/true);
    if (stop < start) {
        stop = start;
    }

    void* new_data_ptr = static_cast<char*>(data_ptr_) +
                         start * strides_[dim] * DtypeUtil::ByteSize(dtype_);
    SizeVector new_shape = shape_;
    SizeVector new_strides = strides_;
    new_shape[dim] = (stop - start + step - 1) / step;
    new_strides[dim] = strides_[dim] * step;
    return Tensor(new_shape, new_strides, new_data_ptr, dtype_, blob_);
}

Tensor Tensor::IndexGet(const std::vector<Tensor>& index_tensors) const {
    AdvancedIndexPreprocessor aip(*this, index_tensors);
    Tensor dst = Tensor(aip.GetOutputShape(), dtype_, GetDevice());
    kernel::IndexGet(aip.GetTensor(), dst, aip.GetIndexTensors(),
                     aip.GetIndexedShape(), aip.GetIndexedStrides());

    return dst;
}

void Tensor::IndexSet(const std::vector<Tensor>& index_tensors,
                      const Tensor& src_tensor) {
    AdvancedIndexPreprocessor aip(*this, index_tensors);
    Tensor pre_processed_dst = aip.GetTensor();
    kernel::IndexSet(src_tensor, pre_processed_dst, aip.GetIndexTensors(),
                     aip.GetIndexedShape(), aip.GetIndexedStrides());
}

Tensor Tensor::Permute(const SizeVector& dims) const {
    // Check dimension size
    if (static_cast<int64_t>(dims.size()) != NumDims()) {
        utility::LogError(
                "Tensor has {} dimensions, but permuntation have {} "
                "dimensions.",
                NumDims(), dims.size());
    }
    int64_t n_dims = NumDims();

    // Check dims are permuntation of [0, 1, 2, ..., n-1]
    std::vector<bool> seen_dims(n_dims, false);
    for (const int64_t& dim : dims) {
        seen_dims[shape_util::WrapDim(dim, n_dims)] = true;
    }
    if (!std::all_of(seen_dims.begin(), seen_dims.end(),
                     [](bool seen) { return seen; })) {
        utility::LogError("Permute dims must be a permuntation from 0 to {}",
                          dims.size() - 1);
    }

    // Map to new shape and strides
    SizeVector new_shape(n_dims);
    SizeVector new_strides(n_dims);
    for (int64_t i = 0; i < n_dims; ++i) {
        int64_t old_dim = shape_util::WrapDim(dims[i], n_dims);
        new_shape[i] = shape_[old_dim];
        new_strides[i] = strides_[old_dim];
    }

    return AsStrided(new_shape, new_strides);
}

Tensor Tensor::AsStrided(const SizeVector& new_shape,
                         const SizeVector& new_strides) const {
    Tensor result(new_shape, new_strides, const_cast<void*>(data_ptr_), dtype_,
                  blob_);
    return result;
}

Tensor Tensor::Transpose(int64_t dim0, int64_t dim1) const {
    int64_t n_dims = NumDims();
    dim0 = shape_util::WrapDim(dim0, n_dims);
    dim1 = shape_util::WrapDim(dim1, n_dims);
    SizeVector dims(n_dims);
    std::iota(dims.begin(), dims.end(), 0);
    dims[dim0] = dim1;
    dims[dim1] = dim0;
    return Permute(dims);
}

Tensor Tensor::T() const {
    int64_t n_dims = NumDims();
    if (n_dims <= 1) {
        return *this;
    } else if (n_dims == 2) {
        return Transpose(0, 1);
    } else {
        utility::LogError(
                "Tensor::T() expects a Tensor with <= 2 dimensions, but the "
                "Tensor as {} dimensions.");
    }
}

Tensor Tensor::Add(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      dtype_, GetDevice());
    kernel::Add(*this, value, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::Add_(const Tensor& value) {
    kernel::Add(*this, value, *this);
    return *this;
}

Tensor Tensor::Sub(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      dtype_, GetDevice());
    kernel::Sub(*this, value, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::Sub_(const Tensor& value) {
    kernel::Sub(*this, value, *this);
    return *this;
}

Tensor Tensor::Mul(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      dtype_, GetDevice());
    kernel::Mul(*this, value, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::Mul_(const Tensor& value) {
    kernel::Mul(*this, value, *this);
    return *this;
}

Tensor Tensor::Div(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      dtype_, GetDevice());
    kernel::Div(*this, value, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::Div_(const Tensor& value) {
    kernel::Div(*this, value, *this);
    return *this;
}

Tensor Tensor::Sum(const SizeVector& dims, bool keepdim) const {
    Tensor dst(shape_util::ReductionShape(shape_, dims, keepdim), dtype_,
               GetDevice());
    kernel::Reduction(*this, dst, dims, keepdim, kernel::ReductionOpCode::Sum);
    return dst;
}

Tensor Tensor::Prod(const SizeVector& dims, bool keepdim) const {
    Tensor dst(shape_util::ReductionShape(shape_, dims, keepdim), dtype_,
               GetDevice());
    kernel::Reduction(*this, dst, dims, keepdim, kernel::ReductionOpCode::Prod);
    return dst;
}

Tensor Tensor::Min(const SizeVector& dims, bool keepdim) const {
    Tensor dst(shape_util::ReductionShape(shape_, dims, keepdim), dtype_,
               GetDevice());
    kernel::Reduction(*this, dst, dims, keepdim, kernel::ReductionOpCode::Min);
    return dst;
}

Tensor Tensor::Max(const SizeVector& dims, bool keepdim) const {
    Tensor dst(shape_util::ReductionShape(shape_, dims, keepdim), dtype_,
               GetDevice());
    kernel::Reduction(*this, dst, dims, keepdim, kernel::ReductionOpCode::Max);
    return dst;
}

Tensor Tensor::ArgMin(const SizeVector& dims) const {
    Tensor dst(shape_util::ReductionShape(shape_, dims, false), Dtype::Int64,
               GetDevice());
    kernel::Reduction(*this, dst, dims, false, kernel::ReductionOpCode::ArgMin);
    return dst;
}

Tensor Tensor::ArgMax(const SizeVector& dims) const {
    Tensor dst(shape_util::ReductionShape(shape_, dims, false), Dtype::Int64,
               GetDevice());
    kernel::Reduction(*this, dst, dims, false, kernel::ReductionOpCode::ArgMax);
    return dst;
}

Tensor Tensor::Sqrt() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Sqrt);
    return dst_tensor;
}

Tensor Tensor::Sqrt_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::Sqrt);
    return *this;
}

Tensor Tensor::Sin() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Sin);
    return dst_tensor;
}

Tensor Tensor::Sin_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::Sin);
    return *this;
}

Tensor Tensor::Cos() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Cos);
    return dst_tensor;
}

Tensor Tensor::Cos_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::Cos);
    return *this;
}

Tensor Tensor::Neg() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Neg);
    return dst_tensor;
}

Tensor Tensor::Neg_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::Neg);
    return *this;
}

Tensor Tensor::Exp() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Exp);
    return dst_tensor;
}

Tensor Tensor::Exp_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::Exp);
    return *this;
}

Tensor Tensor::Abs() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Abs);
    return dst_tensor;
}

Tensor Tensor::Abs_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::Abs);
    return *this;
}

Device Tensor::GetDevice() const {
    if (blob_ == nullptr) {
        utility::LogError("Blob is null, cannot get device");
    }
    return blob_->GetDevice();
}

Tensor Tensor::LogicalNot() const {
    Tensor dst_tensor(shape_, Dtype::Bool, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::LogicalNot);
    return dst_tensor;
}

Tensor Tensor::LogicalNot_() {
    kernel::UnaryEW(*this, *this, kernel::UnaryEWOpCode::LogicalNot);
    return *this;
}

Tensor Tensor::LogicalAnd(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor,
                     kernel::BinaryEWOpCode::LogicalAnd);
    return dst_tensor;
}

Tensor Tensor::LogicalAnd_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::LogicalAnd);
    return *this;
}

Tensor Tensor::LogicalOr(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor,
                     kernel::BinaryEWOpCode::LogicalOr);
    return dst_tensor;
}

Tensor Tensor::LogicalOr_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::LogicalOr);
    return *this;
}

Tensor Tensor::LogicalXor(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor,
                     kernel::BinaryEWOpCode::LogicalXor);
    return dst_tensor;
}

Tensor Tensor::LogicalXor_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::LogicalXor);
    return *this;
}

Tensor Tensor::Gt(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor, kernel::BinaryEWOpCode::Gt);
    return dst_tensor;
}

Tensor Tensor::Gt_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::Gt);
    return *this;
}

Tensor Tensor::Lt(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor, kernel::BinaryEWOpCode::Lt);
    return dst_tensor;
}

Tensor Tensor::Lt_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::Lt);
    return *this;
}

Tensor Tensor::Ge(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor, kernel::BinaryEWOpCode::Ge);
    return dst_tensor;
}

Tensor Tensor::Ge_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::Ge);
    return *this;
}

Tensor Tensor::Le(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor, kernel::BinaryEWOpCode::Le);
    return dst_tensor;
}

Tensor Tensor::Le_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::Le);
    return *this;
}

Tensor Tensor::Eq(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor, kernel::BinaryEWOpCode::Eq);
    return dst_tensor;
}

Tensor Tensor::Eq_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::Eq);
    return *this;
}

Tensor Tensor::Ne(const Tensor& value) const {
    Tensor dst_tensor(shape_util::BroadcastedShape(shape_, value.shape_),
                      Dtype::Bool, GetDevice());
    kernel::BinaryEW(*this, value, dst_tensor, kernel::BinaryEWOpCode::Ne);
    return dst_tensor;
}

Tensor Tensor::Ne_(const Tensor& value) {
    kernel::BinaryEW(*this, value, *this, kernel::BinaryEWOpCode::Ne);
    return *this;
}

}  // namespace open3d
