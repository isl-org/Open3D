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

// High level non-templated hashmap interface for basic usages.

// If BUILD_CUDA_MODULE, link DefaultHashmap.cu that contains everything, and
// disable code inside DefaultHashmap.cpp
// Else, link DefaultHashmap.cpp and disregard DefaultHashmap.cu

#include "open3d/core/hashmap/Hashmap.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {

Hashmap::Hashmap(int64_t init_capacity,
                 const Dtype& dtype_key,
                 const Dtype& dtype_value,
                 const SizeVector& element_shape_key,
                 const SizeVector& element_shape_value,
                 const Device& device)
    : dtype_key_(dtype_key),
      dtype_value_(dtype_value),
      element_shape_key_(element_shape_key),
      element_shape_value_(element_shape_value) {
    if (dtype_key_.GetDtypeCode() == Dtype::DtypeCode::Undefined ||
        dtype_key_.GetDtypeCode() == Dtype::DtypeCode::Undefined) {
        utility::LogError(
                "[Hashmap] DtypeCore::Undefined is not supported for input "
                "key/value.");
    }
    if (element_shape_key_.NumElements() == 0 ||
        element_shape_value_.NumElements() == 0) {
        utility::LogError(
                "[Hashmap] element shape 0 is not supported for input "
                "key/value.");
    }

    device_hashmap_ = CreateDefaultDeviceHashmap(
            std::max(init_capacity / kDefaultElemsPerBucket, int64_t(1)),
            init_capacity,
            dtype_key.ByteSize() * element_shape_key_.NumElements(),
            dtype_value.ByteSize() * element_shape_value_.NumElements(),
            device);
}

void Hashmap::Rehash(int64_t buckets) {
    return device_hashmap_->Rehash(buckets);
}

void Hashmap::Insert(const Tensor& input_keys,
                     const Tensor& input_values,
                     Tensor& output_addrs,
                     Tensor& output_masks) {
    SizeVector input_key_elem_shape(input_keys.GetShape());
    input_key_elem_shape.erase(input_key_elem_shape.begin());
    AssertKeyDtype(input_keys.GetDtype(), input_key_elem_shape);

    SizeVector input_value_elem_shape(input_values.GetShape());
    input_value_elem_shape.erase(input_value_elem_shape.begin());
    AssertValueDtype(input_values.GetDtype(), input_value_elem_shape);

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible key device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    SizeVector value_shape = input_values.GetShape();
    if (value_shape.size() == 0 || value_shape[0] != shape[0]) {
        utility::LogError("[Hashmap]: Invalid value tensor shape");
    }
    if (input_values.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible value device, expected {}, but got {}",
                GetDevice().ToString(), input_values.GetDevice().ToString());
    }

    int64_t count = shape[0];
    output_addrs = Tensor({count}, Dtype::Int32, GetDevice());
    output_masks = Tensor({count}, Dtype::Bool, GetDevice());

    device_hashmap_->Insert(input_keys.GetDataPtr(), input_values.GetDataPtr(),
                            static_cast<addr_t*>(output_addrs.GetDataPtr()),
                            static_cast<bool*>(output_masks.GetDataPtr()),
                            count);
}

void Hashmap::Activate(const Tensor& input_keys,
                       Tensor& output_addrs,
                       Tensor& output_masks) {
    SizeVector input_key_elem_shape(input_keys.GetShape());
    input_key_elem_shape.erase(input_key_elem_shape.begin());
    AssertKeyDtype(input_keys.GetDtype(), input_key_elem_shape);

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    int64_t count = shape[0];

    output_addrs = Tensor({count}, Dtype::Int32, GetDevice());
    output_masks = Tensor({count}, Dtype::Bool, GetDevice());

    device_hashmap_->Activate(input_keys.GetDataPtr(),
                              static_cast<addr_t*>(output_addrs.GetDataPtr()),
                              static_cast<bool*>(output_masks.GetDataPtr()),
                              count);
}

void Hashmap::Find(const Tensor& input_keys,
                   Tensor& output_addrs,
                   Tensor& output_masks) {
    SizeVector input_key_elem_shape(input_keys.GetShape());
    input_key_elem_shape.erase(input_key_elem_shape.begin());
    AssertKeyDtype(input_keys.GetDtype(), input_key_elem_shape);

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    int64_t count = shape[0];

    output_masks = Tensor({count}, Dtype::Bool, GetDevice());
    output_addrs = Tensor({count}, Dtype::Int32, GetDevice());

    device_hashmap_->Find(input_keys.GetDataPtr(),
                          static_cast<addr_t*>(output_addrs.GetDataPtr()),
                          static_cast<bool*>(output_masks.GetDataPtr()), count);
}

void Hashmap::Erase(const Tensor& input_keys, Tensor& output_masks) {
    SizeVector input_key_elem_shape(input_keys.GetShape());
    input_key_elem_shape.erase(input_key_elem_shape.begin());
    AssertKeyDtype(input_keys.GetDtype(), input_key_elem_shape);

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    int64_t count = shape[0];
    output_masks = Tensor({count}, Dtype::Bool, GetDevice());

    device_hashmap_->Erase(input_keys.GetDataPtr(),
                           static_cast<bool*>(output_masks.GetDataPtr()),
                           count);
}

void Hashmap::GetActiveIndices(Tensor& output_addrs) const {
    int64_t count = device_hashmap_->Size();
    output_addrs = Tensor({count}, Dtype::Int32, GetDevice());

    device_hashmap_->GetActiveIndices(
            static_cast<addr_t*>(output_addrs.GetDataPtr()));
}

Hashmap Hashmap::Clone() const { return To(GetDevice(), /*copy=*/true); }

Hashmap Hashmap::To(const Device& device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }

    Hashmap new_hashmap(GetCapacity(), dtype_key_, dtype_value_,
                        element_shape_key_, element_shape_value_, device);

    Tensor keys = GetKeyTensor().To(device, /*copy=*/true);
    Tensor values = GetValueTensor().To(device, /*copy=*/true);

    core::Tensor active_addrs;
    GetActiveIndices(active_addrs);
    core::Tensor active_indices = active_addrs.To(core::Dtype::Int64);

    core::Tensor addrs, masks;
    new_hashmap.Insert(keys.IndexGet({active_indices}),
                       values.IndexGet({active_indices}), addrs, masks);

    return new_hashmap;
}

Hashmap Hashmap::CPU() const { return To(Device("CPU:0"), /*copy=*/false); }

Hashmap Hashmap::CUDA(int device_id) const {
    return To(Device(Device::DeviceType::CUDA, device_id), /*copy=*/false);
}

int64_t Hashmap::Size() const { return device_hashmap_->Size(); }

int64_t Hashmap::GetCapacity() const { return device_hashmap_->GetCapacity(); }
int64_t Hashmap::GetBucketCount() const {
    return device_hashmap_->GetBucketCount();
}
Device Hashmap::GetDevice() const { return device_hashmap_->GetDevice(); }
int64_t Hashmap::GetKeyBytesize() const {
    return device_hashmap_->GetKeyBytesize();
}
int64_t Hashmap::GetValueBytesize() const {
    return device_hashmap_->GetValueBytesize();
}

Tensor& Hashmap::GetKeyBuffer() const {
    return device_hashmap_->GetKeyBuffer();
}
Tensor& Hashmap::GetValueBuffer() const {
    return device_hashmap_->GetValueBuffer();
}

Tensor Hashmap::GetKeyTensor() const {
    int64_t capacity = GetCapacity();
    SizeVector key_shape = element_shape_key_;
    key_shape.insert(key_shape.begin(), capacity);
    return Tensor(key_shape, shape_util::DefaultStrides(key_shape),
                  GetKeyBuffer().GetDataPtr(), dtype_key_,
                  GetKeyBuffer().GetBlob());
}

Tensor Hashmap::GetValueTensor() const {
    int64_t capacity = GetCapacity();
    SizeVector value_shape = element_shape_value_;
    value_shape.insert(value_shape.begin(), capacity);
    return Tensor(value_shape, shape_util::DefaultStrides(value_shape),
                  GetValueBuffer().GetDataPtr(), dtype_value_,
                  GetValueBuffer().GetBlob());
}

/// Return number of elems per bucket.
/// High performance not required, so directly returns a vector.
std::vector<int64_t> Hashmap::BucketSizes() const {
    return device_hashmap_->BucketSizes();
};

/// Return size / bucket_count.
float Hashmap::LoadFactor() const { return device_hashmap_->LoadFactor(); }

void Hashmap::AssertKeyDtype(const Dtype& dtype_key,
                             const SizeVector& element_shape_key) const {
    int64_t elem_byte_size =
            dtype_key.ByteSize() * element_shape_key.NumElements();
    int64_t stored_elem_byte_size =
            dtype_key_.ByteSize() * element_shape_key_.NumElements();
    if (elem_byte_size != stored_elem_byte_size) {
        utility::LogError(
                "[Hashmap] Inconsistent element-wise key byte size, expected "
                "{}, but got {}",
                stored_elem_byte_size, elem_byte_size);
    }
}

void Hashmap::AssertValueDtype(const Dtype& dtype_value,
                               const SizeVector& element_shape_value) const {
    int64_t elem_byte_size =
            dtype_value.ByteSize() * element_shape_value.NumElements();
    int64_t stored_elem_byte_size =
            dtype_value_.ByteSize() * element_shape_value_.NumElements();
    if (elem_byte_size != stored_elem_byte_size) {
        utility::LogError(
                "[Hashmap] Inconsistent element-wise value byte size, expected "
                "{}, but got {}",
                stored_elem_byte_size, elem_byte_size);
    }
}

}  // namespace core
}  // namespace open3d
