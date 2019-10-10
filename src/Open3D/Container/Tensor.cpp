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

#include "Open3D/Container/Tensor.h"

#include <sstream>

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

Tensor Tensor::CopyTo(const Device& device) const {
    // TODO: contiguous transform can happen together with copy in kernel
    Tensor contiguous_tensor = Contiguous();
    Tensor dst_tensor(shape_, dtype_, device);
    MemoryManager::Memcpy(dst_tensor.GetDataPtr(), dst_tensor.GetDevice(),
                          contiguous_tensor.GetDataPtr(),
                          contiguous_tensor.GetDevice(),
                          shape_.NumElements() * DtypeUtil::ByteSize(dtype_));
    return dst_tensor;
}

Tensor Tensor::CloneTo(const Device& device) const {
    auto new_blob = std::make_shared<Blob>(blob_->byte_size_, device);
    MemoryManager::MemcpyBlob(new_blob, blob_);
    size_t data_offset =
            static_cast<uint8_t*>(data_ptr_) - static_cast<uint8_t*>(blob_->v_);
    void* new_data_ptr = static_cast<uint8_t*>(new_blob->v_) + data_offset;
    return Tensor(shape_, strides_, new_data_ptr, dtype_, device, new_blob);
}

Tensor Tensor::Contiguous() const {
    if (IsContiguous()) {
        // Returns a shallow copy of the current Tensor
        return Tensor(shape_, strides_, data_ptr_, dtype_, device_, blob_);
    } else {
        // TODO: Make both CPU/CUDA kernels an OP, with registration mecanism
        // TOOD: Consider making a Tensor accessor class
        if (device_.device_type_ == Device::DeviceType::CUDA) {
            // TODO: write a CUDA Kernel
            Tensor cpu_clone = CloneTo(Device("CPU:0"));
            Tensor cpu_contiguous = cpu_clone.Contiguous();
            Tensor cuda_contiguous = cpu_contiguous.CloneTo(device_);
            return cuda_contiguous;
        } else if (device_.device_type_ == Device::DeviceType::CPU) {
            Tensor dst_tensor(shape_, dtype_, device_);
            // int64_t to avoid MSVC openmp error
            int64_t num_elements = static_cast<int64_t>(shape_.NumElements());
            size_t num_dims = shape_.size();
            SizeVector default_strides = DefaultStrides(shape_);
            size_t element_byte_size = DtypeUtil::ByteSize(dtype_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t dst_offset = 0; dst_offset < num_elements;
                 dst_offset++) {
                size_t ind = static_cast<size_t>(dst_offset);
                SizeVector indices(shape_.size());
                size_t src_offset = 0;
                for (size_t dim = 0; dim < num_dims; dim++) {
                    src_offset += ind / default_strides[dim] * strides_[dim];
                    ind = ind % default_strides[dim];
                }
                void* src_ptr = static_cast<uint8_t*>(data_ptr_) +
                                src_offset * element_byte_size;
                void* dst_ptr = static_cast<uint8_t*>(dst_tensor.GetDataPtr()) +
                                dst_offset * element_byte_size;
                MemoryManager::Memcpy(dst_ptr, dst_tensor.GetDevice(),
                                      const_cast<const void*>(src_ptr),
                                      GetDevice(), element_byte_size);
            }
            return dst_tensor;
        } else {
            utility::LogFatal("Unknown device\n");
        }
    }
}

SizeVector Tensor::DefaultStrides(const SizeVector& shape) {
    SizeVector strides(shape.size());
    size_t stride_size = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride_size;
        // Handles 0-sized dimensions
        stride_size *= std::max<size_t>(shape[i - 1], 1);
    }
    return strides;
}

std::string Tensor::ToString(bool with_suffix,
                             const std::string& indent) const {
    std::ostringstream rc;

    if (device_.device_type_ == Device::DeviceType::CUDA) {
        // Copy to CPU for printing
        // TODO: improve Contiguous() so that only the used part is copied
        Tensor host_tensor = CopyTo(Device("CPU:0"));
        rc << host_tensor.ToString(false, "");
    } else if (!IsContiguous()) {
        // Copy to Contiguous() buffer for printing
        // TODO: if we implement an Accessor class, can maybe avoid this
        Tensor contiguous_tensor = Contiguous();
        rc << contiguous_tensor.ToString(false, "");
    } else {
        if (shape_.size() == 0) {
            rc << indent;
            rc << ScalarPtrToString(data_ptr_);
        } else if (shape_.size() == 1) {
            const uint8_t* ptr = static_cast<const uint8_t*>(data_ptr_);
            rc << "[";
            std::string delim = "";
            size_t element_byte_size = DtypeUtil::ByteSize(dtype_);
            for (size_t i = 0; i < shape_.NumElements(); ++i) {
                rc << delim << ScalarPtrToString(ptr);
                delim = " ";
                ptr += element_byte_size;
            }
            rc << "]";
        } else {
            rc << "[";
            std::string delim = "";
            std::string child_indent = "";
            for (size_t i = 0; i < shape_[0]; ++i) {
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
                          DtypeUtil::ToString(dtype_), device_.ToString(),
                          data_ptr_);
    }
    return rc.str();
}

std::string Tensor::ScalarPtrToString(const void* ptr) const {
    std::string str = "";
    switch (dtype_) {
        case Dtype::Float32:
            str = fmt::format("{}", *static_cast<const float*>(ptr));
            break;
        case Dtype::Float64:
            str = fmt::format("{}", *static_cast<const double*>(ptr));
            break;
        case Dtype::Int32:
            str = fmt::format("{}", *static_cast<const int32_t*>(ptr));
            break;
        case Dtype::Int64:
            str = fmt::format("{}", *static_cast<const int64_t*>(ptr));
            break;
        case Dtype::UInt8:
            str = fmt::format("{}", *static_cast<const uint8_t*>(ptr));
            break;
        default:
            utility::LogFatal("Unsupported data type\n");
    }
    return str;
}

Tensor Tensor::operator[](size_t i) const {
    if (shape_.size() == 0) {
        utility::LogFatal("Tensor has shape (), cannot be indexed.\n");
    }
    if (i < 0) {
        utility::LogFatal(
                "Only non-ngegative index is supported, but {} < 0.\n", i);
    }
    if (i >= shape_[0]) {
        utility::LogFatal("Index {} is out of bounds for axis of length {}.\n",
                          i, shape_[0]);
    }
    if (shape_.size() != strides_.size()) {
        utility::LogFatal(
                "Internal error, shape and strides dimension mismatch {} != "
                "{}\n",
                shape_.size(), strides_.size());
    }
    SizeVector new_shape(shape_.begin() + 1, shape_.end());
    SizeVector new_stride(strides_.begin() + 1, strides_.end());
    void* new_data_ptr = static_cast<uint8_t*>(data_ptr_) +
                         strides_[0] * DtypeUtil::ByteSize(dtype_) * i;
    return Tensor(new_shape, new_stride, new_data_ptr, dtype_, device_, blob_);
}

Tensor Tensor::Slice(size_t dim, size_t start, size_t stop, size_t step) const {
    if (shape_.size() == 0) {
        utility::LogFatal("Slice cannot be applied to 0-dim Tensor\n");
    }
    if (dim < 0 || dim >= shape_.size()) {
        utility::LogFatal(
                "Dim {} is out of bound for SizeVector of length {}\n", dim,
                shape_.size());
    }
    // TODO: support negative step sizes
    if (step == 0) {
        utility::LogFatal("Step size cannot be 0\n");
    }
    // TODO: support wrap-around start/stop index
    if (start < 0 || start >= shape_[dim]) {
        utility::LogFatal("Index {} is out of bounds for axis of length {}.\n",
                          start, shape_[dim]);
    }
    // The stop index is non-inclusive
    if (stop < 0 || stop > shape_[dim]) {
        utility::LogFatal("Index {} is out of bounds for axis of length {}.\n",
                          stop, shape_[dim]);
    }
    if (stop < start) {
        stop = start;
    }

    void* new_data_ptr = static_cast<uint8_t*>(data_ptr_) +
                         start * strides_[dim] * DtypeUtil::ByteSize(dtype_);
    SizeVector new_shape = shape_;
    SizeVector new_strides = strides_;
    new_shape[dim] = (stop - start + step - 1) / step;
    new_strides[dim] = strides_[dim] * step;
    return Tensor(new_shape, new_strides, new_data_ptr, dtype_, device_, blob_);
}

}  // namespace open3d
