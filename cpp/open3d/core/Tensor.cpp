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

#include "open3d/core/Tensor.h"

#include <sstream>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Blob.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/NumpyIO.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/TensorKey.h"
#include "open3d/core/kernel/Arange.h"
#include "open3d/core/kernel/Kernel.h"
#include "open3d/core/linalg/Inverse.h"
#include "open3d/core/linalg/LeastSquares.h"
#include "open3d/core/linalg/Matmul.h"
#include "open3d/core/linalg/SVD.h"
#include "open3d/core/linalg/Solve.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {

/// Open3D DLPack Tensor manager.
class Open3DDLManagedTensor {
private:
    Open3DDLManagedTensor(const Tensor& o3d_tensor) {
        o3d_tensor_ = o3d_tensor;

        // Prepare dl_device_type
        DLDeviceType dl_device_type;
        Device device = o3d_tensor_.GetDevice();
        switch (device.GetType()) {
            case Device::DeviceType::CPU:
                dl_device_type = DLDeviceType::kDLCPU;
                break;
            case Device::DeviceType::CUDA:
                dl_device_type = DLDeviceType::kDLGPU;
                break;
            default:
                utility::LogError("ToDLPack: unsupported device type {}",
                                  device.ToString());
        }

        // Prepare dl_context
        DLContext dl_context;
        dl_context.device_type = dl_device_type;
        dl_context.device_id = device.GetID();

        // Prepare dl_data_type
        DLDataType dl_data_type;
        Dtype dtype = o3d_tensor_.GetDtype();

        if (dtype == Dtype::Float32) {
            dl_data_type.code = DLDataTypeCode::kDLFloat;
        } else if (dtype == Dtype::Float64) {
            dl_data_type.code = DLDataTypeCode::kDLFloat;
        } else if (dtype == Dtype::Int32) {
            dl_data_type.code = DLDataTypeCode::kDLInt;
        } else if (dtype == Dtype::Int64) {
            dl_data_type.code = DLDataTypeCode::kDLInt;
        } else if (dtype == Dtype::UInt8) {
            dl_data_type.code = DLDataTypeCode::kDLUInt;
        } else if (dtype == Dtype::UInt16) {
            dl_data_type.code = DLDataTypeCode::kDLUInt;
        } else {
            utility::LogError("Unsupported data type");
        }

        dl_data_type.bits = static_cast<uint8_t>(dtype.ByteSize() * 8);
        dl_data_type.lanes = 1;

        // Prepare dl_tensor, this uses dl_device_type, dl_context and
        // dl_data_type prepared above.
        DLTensor dl_tensor;
        // Not Blob's data pointer.
        dl_tensor.data = const_cast<void*>(o3d_tensor_.GetDataPtr());
        dl_tensor.ctx = dl_context;
        dl_tensor.ndim = static_cast<int>(o3d_tensor_.GetShape().size());
        dl_tensor.dtype = dl_data_type;
        // The shape pointer is alive for the lifetime of Open3DDLManagedTensor.
        dl_tensor.shape =
                const_cast<int64_t*>(o3d_tensor_.GetShapeRef().data());
        // The strides pointer is alive for the lifetime of
        // Open3DDLManagedTensor.
        dl_tensor.strides =
                const_cast<int64_t*>(o3d_tensor_.GetStridesRef().data());
        dl_tensor.byte_offset = 0;

        dl_managed_tensor_.manager_ctx = this;
        dl_managed_tensor_.deleter = &Open3DDLManagedTensor::Deleter;
        dl_managed_tensor_.dl_tensor = dl_tensor;
    }

    Tensor o3d_tensor_;
    DLManagedTensor dl_managed_tensor_;

public:
    /// `DLManagedTensor* dmlt` is destroyed by calling `dmlt->deleter(dmlt)`.
    /// The destruction happens when the DLPack python object goes out of scope,
    /// and ultimately it decreases the reference count to the actual data
    /// buffer (i.e. `dmlt.manager_ctx->o3d_tensor_.GetBlob()`) by 1.
    static DLManagedTensor* Create(const Tensor& o3d_tensor) {
        Open3DDLManagedTensor* o3d_dl_tensor =
                new Open3DDLManagedTensor(o3d_tensor);
        return &o3d_dl_tensor->dl_managed_tensor_;
    }

    static void Deleter(DLManagedTensor* arg) {
        delete static_cast<Open3DDLManagedTensor*>(arg->manager_ctx);
    }
};

// Equivalent to `Tensor& operator=(const Tensor& other) & = default;`.
// Manual implentaiton is need to avoid MSVC bug (error C2580:  multiple
// versions of a defaulted special member functions are not allowed.)
Tensor& Tensor::operator=(const Tensor& other) & {
    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    blob_ = other.blob_;
    data_ptr_ = other.data_ptr_;
    return *this;
}

// Equivalent to `Tensor& operator=(Tensor&& other) & = default;`.
// Manual implentaiton is need to avoid MSVC bug (error C2580:  multiple
// versions of a defaulted special member functions are not allowed.)
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

Tensor Tensor::Empty(const SizeVector& shape,
                     Dtype dtype,
                     const Device& device) {
    return Tensor(shape, dtype, device);
}

Tensor Tensor::Zeros(const SizeVector& shape,
                     Dtype dtype,
                     const Device& device) {
    return Full(shape, 0, dtype, device);
}

Tensor Tensor::Ones(const SizeVector& shape,
                    Dtype dtype,
                    const Device& device) {
    return Full(shape, 1, dtype, device);
}

Tensor Tensor::Eye(int64_t n, Dtype dtype, const Device& device) {
    Tensor eye = Tensor::Zeros({n, n}, dtype, device);
    eye.AsStrided({n}, {eye.strides_[0] + eye.strides_[1]}).Fill(1);
    return eye;
}

Tensor Tensor::Diag(const Tensor& input) {
    const SizeVector& shape = input.GetShape();
    if (shape.size() != 1) {
        utility::LogError("Input tensor must be 1D, but got shape {}.",
                          input.shape_.ToString());
    }
    int64_t n = shape[0];
    Tensor diag = Tensor::Zeros({n, n}, input.GetDtype(), input.GetDevice());
    diag.AsStrided({n}, {diag.strides_[0] + diag.strides_[1]}) = input;
    return diag;
}

Tensor Tensor::Arange(Scalar start,
                      Scalar stop,
                      Scalar step,
                      Dtype dtype,
                      const Device& device) {
    start.AssertSameScalarType(stop,
                               "start must have the same scalar type as stop.");
    start.AssertSameScalarType(step,
                               "start must have the same scalar type as step.");

    if (step.Equal(0)) {
        utility::LogError("Step cannot be 0.");
    }
    if (stop.Equal(start)) {
        return Tensor({0}, dtype, device);
    }

    Tensor t_start;
    Tensor t_stop;
    Tensor t_step;
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        t_start = Tensor::Full({}, start.To<scalar_t>(), dtype, device);
        t_stop = Tensor::Full({}, stop.To<scalar_t>(), dtype, device);
        t_step = Tensor::Full({}, step.To<scalar_t>(), dtype, device);
    });

    return kernel::Arange(t_start, t_stop, t_step);
}

Tensor Tensor::GetItem(const TensorKey& tk) const {
    if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
        return IndexExtract(0, tk.GetIndex());
    } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
        TensorKey tk_new = tk.InstantiateDimSize(shape_[0]);
        return Slice(0, tk_new.GetStart(), tk_new.GetStop(), tk_new.GetStep());
    } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
        return IndexGet({tk.GetIndexTensor()});
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
                index_tensors.push_back(tk.GetIndexTensor());
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
            TensorKey tk_new = tk.InstantiateDimSize(t.shape_[slice_dim]);
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
        IndexSet({tk.GetIndexTensor()}, value);
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
                index_tensors.push_back(tk.GetIndexTensor());
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
    strides_ = shape_util::DefaultStrides(shape_);
    dtype_ = other.dtype_;
    blob_ = std::make_shared<Blob>(shape_.NumElements() * dtype_.ByteSize(),
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
            shape_util::Restride(shape_, strides_, inferred_dst_shape);
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
            shape_util::Restride(shape_, strides_, inferred_dst_shape);
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

Tensor Tensor::To(Dtype dtype, bool copy) const {
    if (!copy && dtype_ == dtype) {
        return *this;
    }
    // We only support scalar type conversion.
    if (dtype_.IsObject() || dtype.IsObject()) {
        utility::LogError("Cannot cast type from {} to {}.", dtype_.ToString(),
                          dtype.ToString());
    }
    Tensor dst_tensor(shape_, dtype, GetDevice());
    kernel::Copy(*this, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::To(const Device& device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }
    Tensor dst_tensor(shape_, dtype_, device);
    kernel::Copy(*this, dst_tensor);
    return dst_tensor;
}

Tensor Tensor::To(const Device& device, Dtype dtype, bool copy) const {
    Tensor dst_tensor = To(dtype, copy);
    dst_tensor = dst_tensor.To(device, copy);
    return dst_tensor;
}

void Tensor::CopyFrom(const Tensor& other) { AsRvalue() = other; }

Tensor Tensor::Contiguous() const {
    if (IsContiguous()) {
        return *this;
    } else {
        return To(GetDevice(), /*copy=*/true);
    }
}

std::string Tensor::ToString(bool with_suffix,
                             const std::string& indent) const {
    std::ostringstream rc;
    if (GetDevice().GetType() == Device::DeviceType::CUDA || !IsContiguous()) {
        Tensor host_contiguous_tensor = Contiguous().To(Device("CPU:0"));
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
            int64_t element_byte_size = dtype_.ByteSize();
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
                          dtype_.ToString(), GetDevice().ToString(), data_ptr_);
    }
    return rc.str();
}

std::string Tensor::ScalarPtrToString(const void* ptr) const {
    std::string str = "";
    if (dtype_ == Dtype::Bool) {
        str = *static_cast<const unsigned char*>(ptr) ? "True" : "False";
    } else if (dtype_.IsObject()) {
        str = fmt::format("{}", fmt::ptr(ptr));
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
    dim = shape_util::WrapDim(dim, NumDims());
    idx = shape_util::WrapDim(idx, shape_[dim]);

    SizeVector new_shape(shape_);
    new_shape.erase(new_shape.begin() + dim);
    SizeVector new_strides(strides_);
    new_strides.erase(new_strides.begin() + dim);
    void* new_data_ptr = static_cast<char*>(data_ptr_) +
                         strides_[dim] * dtype_.ByteSize() * idx;
    return Tensor(new_shape, new_strides, new_data_ptr, dtype_, blob_);
}

Tensor Tensor::Slice(int64_t dim,
                     int64_t start,
                     int64_t stop,
                     int64_t step) const {
    if (shape_.size() == 0) {
        utility::LogError("Slice cannot be applied to 0-dim Tensor.");
    }
    dim = shape_util::WrapDim(dim, NumDims());
    if (dim < 0 || dim >= static_cast<int64_t>(shape_.size())) {
        utility::LogError("Dim {} is out of bound for SizeVector of length {}.",
                          dim, shape_.size());
    }
    if (step == 0) {
        utility::LogError("Step size cannot be 0.");
    } else if (step < 0) {
        // TODO: support negative step sizes
        utility::LogError("Step size cannot be < 0.");
    }

    // Wrap start. Out-of-range slice is valid and produces empty Tensor.
    if (start < 0) {
        start += shape_[dim];
    }
    if (start < 0) {
        start = 0;
    } else if (start >= shape_[dim]) {
        start = shape_[dim];
    }

    // Wrap stop. Out-of-range slice is valid and produces empty Tensor.
    if (stop < 0) {
        stop += shape_[dim];
    }
    if (stop < start) {
        stop = start;
    } else if (stop >= shape_[dim]) {
        stop = shape_[dim];
    }

    void* new_data_ptr = static_cast<char*>(data_ptr_) +
                         start * strides_[dim] * dtype_.ByteSize();
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

double Tensor::Det() const {
    // TODO: Create a proper op for Determinant.
    this->AssertShape({3, 3});
    this->AssertDtype(core::Dtype::Float32);
    core::Tensor D_ = this->Clone();
    D_[0][0] = D_[0][0] * (D_[1][1] * D_[2][2] - D_[1][2] * D_[2][1]) -
               D_[0][1] * (D_[1][0] * D_[2][2] - D_[2][0] * D_[1][2]) +
               D_[0][2] * (D_[1][0] * D_[2][1] - D_[2][0] * D_[1][1]);
    return static_cast<double>(D_[0][0].Item<float>());
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

Tensor Tensor::Mean(const SizeVector& dims, bool keepdim) const {
    if (dtype_ != Dtype::Float32 && dtype_ != Dtype::Float64) {
        utility::LogError(
                "Can only compute mean for Float32 or Float64, got {} instead.",
                dtype_.ToString());
    }

    // Following Numpy's semantics, reduction on 0-sized Tensor will result in
    // NaNs and a warning. A straightforward method is used now. Later it can be
    // extended to handle overflow and underflow in a better way.
    if (NumElements() == 0) {
        utility::LogWarning("Computing mean of 0-sized Tensor.");
    }
    Tensor sum = Sum(dims, keepdim);
    double factor = static_cast<double>(sum.NumElements()) / NumElements();
    return sum * factor;
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

Tensor Tensor::Floor() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Floor);
    return dst_tensor;
}

Tensor Tensor::Ceil() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Ceil);
    return dst_tensor;
}

Tensor Tensor::Round() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Round);
    return dst_tensor;
}

Tensor Tensor::Trunc() const {
    Tensor dst_tensor(shape_, dtype_, GetDevice());
    kernel::UnaryEW(*this, dst_tensor, kernel::UnaryEWOpCode::Trunc);
    return dst_tensor;
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

std::vector<Tensor> Tensor::NonZeroNumpy() const {
    Tensor result = kernel::NonZero(*this);
    std::vector<Tensor> results;
    for (int64_t dim = 0; dim < NumDims(); dim++) {
        results.push_back(result[dim].Clone());
    }
    return results;
}

Tensor Tensor::NonZero() const { return kernel::NonZero(*this); }

bool Tensor::IsNonZero() const {
    if (shape_.NumElements() != 1) {
        utility::LogError(
                "Tensor must have exactly one element to be evaluated as "
                "boolean.");
    }
    bool rc = false;
    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype_, [&]() {
        rc = Item<scalar_t>() != static_cast<scalar_t>(0);
    });
    return rc;
}

bool Tensor::All() const {
    Tensor dst({}, dtype_, GetDevice());
    kernel::Reduction(*this, dst, shape_util::Iota(NumDims()), false,
                      kernel::ReductionOpCode::All);
    return dst.Item<bool>();
}

bool Tensor::Any() const {
    Tensor dst({}, dtype_, GetDevice());
    kernel::Reduction(*this, dst, shape_util::Iota(NumDims()), false,
                      kernel::ReductionOpCode::Any);
    return dst.Item<bool>();
}

DLManagedTensor* Tensor::ToDLPack() const {
    return Open3DDLManagedTensor::Create(*this);
}

Tensor Tensor::FromDLPack(const DLManagedTensor* src) {
    Device device;
    switch (src->dl_tensor.ctx.device_type) {
        case DLDeviceType::kDLCPU:
            device = Device("CPU", src->dl_tensor.ctx.device_id);
            break;
        case DLDeviceType::kDLGPU:
            device = Device("CUDA", src->dl_tensor.ctx.device_id);
            break;
        default:
            utility::LogError("Unsupported device_type {}",
                              src->dl_tensor.ctx.device_type);
    }

    Dtype dtype;
    if (src->dl_tensor.dtype.lanes != 1) {
        utility::LogError("Only supports lanes == 1, but lanes == {}",
                          src->dl_tensor.dtype.lanes);
    }
    switch (src->dl_tensor.dtype.code) {
        case DLDataTypeCode::kDLUInt:
            switch (src->dl_tensor.dtype.bits) {
                case 8:
                    dtype = Dtype::UInt8;
                    break;
                case 16:
                    dtype = Dtype::UInt16;
                    break;
                default:
                    utility::LogError("Unsupported kDLUInt bits {}",
                                      src->dl_tensor.dtype.bits);
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (src->dl_tensor.dtype.bits) {
                case 32:
                    dtype = Dtype::Int32;
                    break;
                case 64:
                    dtype = Dtype::Int64;
                    break;
                default:
                    utility::LogError("Unsupported kDLInt bits {}",
                                      src->dl_tensor.dtype.bits);
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (src->dl_tensor.dtype.bits) {
                case 32:
                    dtype = Dtype::Float32;
                    break;
                case 64:
                    dtype = Dtype::Float64;
                    break;
                default:
                    utility::LogError("Unsupported kDLFloat bits {}",
                                      src->dl_tensor.dtype.bits);
            }
            break;
        default:
            utility::LogError("Unsupported dtype code {}",
                              src->dl_tensor.dtype.code);
    }

    // Open3D Blob's expects an std::function<void(void*)> deleter.
    auto deleter = [src](void* dummy) -> void {
        if (src->deleter != nullptr) {
            src->deleter(const_cast<DLManagedTensor*>(src));
        }
    };

    SizeVector shape(src->dl_tensor.shape,
                     src->dl_tensor.shape + src->dl_tensor.ndim);

    SizeVector strides;
    if (src->dl_tensor.strides == nullptr) {
        strides = shape_util::DefaultStrides(shape);
    } else {
        strides = SizeVector(src->dl_tensor.strides,
                             src->dl_tensor.strides + src->dl_tensor.ndim);
    }

    auto blob = std::make_shared<Blob>(device, src->dl_tensor.data, deleter);

    // src->dl_tensor.byte_offset is ignored in PyTorch and MXNet, but
    // according to dlpack.h, we added the offset here.
    return Tensor(shape, strides,
                  reinterpret_cast<char*>(blob->GetDataPtr()) +
                          src->dl_tensor.byte_offset,
                  dtype, blob);
}

void Tensor::Save(const std::string& file_name) const {
    NumpyArray(*this).Save(file_name);
}

Tensor Tensor::Load(const std::string& file_name) {
    return NumpyArray::Load(file_name).ToTensor();
}

bool Tensor::AllClose(const Tensor& other, double rtol, double atol) const {
    // TODO: support nan;
    return IsClose(other, rtol, atol).All();
}

Tensor Tensor::IsClose(const Tensor& other, double rtol, double atol) const {
    if (GetDevice() != other.GetDevice()) {
        utility::LogError("Device mismatch {} != {}.", GetDevice().ToString(),
                          other.GetDevice().ToString());
    }
    if (dtype_ != other.dtype_) {
        utility::LogError("Dtype mismatch {} != {}.", dtype_.ToString(),
                          other.dtype_.ToString());
    }
    if (shape_ != other.shape_) {
        utility::LogError("Shape mismatch {} != {}.", shape_, other.shape_);
    }

    Tensor lhs = this->To(Dtype::Float64);
    Tensor rhs = other.To(Dtype::Float64);
    Tensor actual_error = (lhs - rhs).Abs();
    Tensor max_error = atol + rtol * rhs.Abs();
    return actual_error <= max_error;
}

bool Tensor::IsSame(const Tensor& other) const {
    return blob_ == other.blob_ && shape_ == other.shape_ &&
           strides_ == other.strides_ && data_ptr_ == other.data_ptr_ &&
           dtype_ == other.dtype_;
}

void Tensor::AssertShape(const SizeVector& expected_shape,
                         const std::string& error_msg) const {
    if (shape_ != expected_shape) {
        if (error_msg.empty()) {
            utility::LogError(
                    "Tensor has shape {}, but it is expected to be {}.", shape_,
                    expected_shape);
        } else {
            utility::LogError(
                    "Tensor has shape {}, but it is expected to be {}: {}",
                    shape_, expected_shape, error_msg);
        }
    }
}

void Tensor::AssertShapeCompatible(const DynamicSizeVector& expected_shape,
                                   const std::string& error_msg) const {
    GetShape().AssertCompatible(expected_shape, error_msg);
}

void Tensor::AssertDevice(const Device& expected_device,
                          const std::string& error_msg) const {
    if (GetDevice() != expected_device) {
        if (error_msg.empty()) {
            utility::LogError("Tensor has device {}, but is expected to be {}",
                              GetDevice().ToString(),
                              expected_device.ToString());
        } else {
            utility::LogError(
                    "Tensor has device {}, but is expected to be {}: {}",
                    GetDevice().ToString(), expected_device.ToString(),
                    error_msg);
        }
    }
}

void Tensor::AssertDtype(const Dtype& expected_dtype,
                         const std::string& error_msg) const {
    if (GetDtype() != expected_dtype) {
        if (error_msg.empty()) {
            utility::LogError("Tensor has dtype {}, but is expected to be {}.",
                              GetDtype().ToString(), expected_dtype.ToString());
        } else {
            utility::LogError(
                    "Tensor has dtype {}, but is expected to be {}: {}",
                    GetDtype().ToString(), expected_dtype.ToString(),
                    error_msg);
        }
    }
}

Tensor Tensor::Matmul(const Tensor& rhs) const {
    Tensor output;
    core::Matmul(*this, rhs, output);
    return output;
}

Tensor Tensor::Solve(const Tensor& rhs) const {
    Tensor output;
    core::Solve(*this, rhs, output);
    return output;
};

Tensor Tensor::LeastSquares(const Tensor& rhs) const {
    Tensor output;
    core::LeastSquares(*this, rhs, output);
    return output;
};

Tensor Tensor::Inverse() const {
    Tensor output;
    core::Inverse(*this, output);
    return output;
}

std::tuple<Tensor, Tensor, Tensor> Tensor::SVD() const {
    Tensor U, S, VT;
    core::SVD(*this, U, S, VT);
    return std::tie(U, S, VT);
}

}  // namespace core
}  // namespace open3d
