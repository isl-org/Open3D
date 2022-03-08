// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/hashmap/HashMap.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/t/io/HashMapIO.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

HashMap::HashMap(int64_t init_capacity,
                 const Dtype& key_dtype,
                 const SizeVector& key_element_shape,
                 const Dtype& value_dtype,
                 const SizeVector& value_element_shape,
                 const Device& device,
                 const HashBackendType& backend)
    : key_dtype_(key_dtype),
      key_element_shape_(key_element_shape),
      dtypes_value_({value_dtype}),
      element_shapes_value_({value_element_shape}) {
    Init(init_capacity, device, backend);
}

HashMap::HashMap(int64_t init_capacity,
                 const Dtype& key_dtype,
                 const SizeVector& key_element_shape,
                 const std::vector<Dtype>& dtypes_value,
                 const std::vector<SizeVector>& element_shapes_value,
                 const Device& device,
                 const HashBackendType& backend)
    : key_dtype_(key_dtype),
      key_element_shape_(key_element_shape),
      dtypes_value_(dtypes_value),
      element_shapes_value_(element_shapes_value) {
    Init(init_capacity, device, backend);
}

void HashMap::Reserve(int64_t capacity) {
    int64_t count = Size();
    if (capacity <= count) {
        utility::LogDebug("Target capacity smaller then current size, abort.");
        return;
    }

    Tensor active_keys;
    std::vector<Tensor> active_values;

    if (count > 0) {
        Tensor active_buf_indices = GetActiveIndices();
        Tensor active_indices = active_buf_indices.To(core::Int64);

        active_keys = GetKeyTensor().IndexGet({active_indices});
        auto value_buffers = GetValueTensors();
        for (auto& value_buffer : value_buffers) {
            active_values.emplace_back(value_buffer.IndexGet({active_indices}));
        }
    }

    device_hashmap_->Free();
    device_hashmap_->Allocate(capacity);
    device_hashmap_->Reserve(capacity);

    if (count > 0) {
        Tensor output_buf_indices, output_masks;
        InsertImpl(active_keys, active_values, output_buf_indices,
                   output_masks);
    }
}

std::pair<Tensor, Tensor> HashMap::Insert(const Tensor& input_keys,
                                          const Tensor& input_values) {
    Tensor output_buf_indices, output_masks;
    Insert(input_keys, input_values, output_buf_indices, output_masks);
    return std::make_pair(output_buf_indices, output_masks);
}

std::pair<Tensor, Tensor> HashMap::Insert(
        const Tensor& input_keys, const std::vector<Tensor>& input_values_soa) {
    Tensor output_buf_indices, output_masks;
    Insert(input_keys, input_values_soa, output_buf_indices, output_masks);
    return std::make_pair(output_buf_indices, output_masks);
}

std::pair<Tensor, Tensor> HashMap::Activate(const Tensor& input_keys) {
    Tensor output_buf_indices, output_masks;
    Activate(input_keys, output_buf_indices, output_masks);
    return std::make_pair(output_buf_indices, output_masks);
}

std::pair<Tensor, Tensor> HashMap::Find(const Tensor& input_keys) {
    Tensor output_buf_indices, output_masks;
    Find(input_keys, output_buf_indices, output_masks);
    return std::make_pair(output_buf_indices, output_masks);
}

Tensor HashMap::Erase(const Tensor& input_keys) {
    Tensor output_masks;
    Erase(input_keys, output_masks);
    return output_masks;
}

Tensor HashMap::GetActiveIndices() const {
    Tensor output_buf_indices;
    GetActiveIndices(output_buf_indices);
    return output_buf_indices;
}

void HashMap::InsertImpl(const Tensor& input_keys,
                         const std::vector<Tensor>& input_values_soa,
                         Tensor& output_buf_indices,
                         Tensor& output_masks,
                         bool is_activate_op) {
    CheckKeyCompatibility(input_keys);
    if (!is_activate_op) {
        CheckKeyValueLengthCompatibility(input_keys, input_values_soa);
        CheckValueCompatibility(input_values_soa);
    }

    int64_t length = input_keys.GetLength();
    PrepareIndicesOutput(output_buf_indices, length);
    PrepareMasksOutput(output_masks, length);

    std::vector<const void*> input_values_ptrs;
    for (const auto& input_value : input_values_soa) {
        input_values_ptrs.push_back(input_value.GetDataPtr());
    }

    device_hashmap_->Insert(
            input_keys.GetDataPtr(), input_values_ptrs,
            static_cast<buf_index_t*>(output_buf_indices.GetDataPtr()),
            output_masks.GetDataPtr<bool>(), length);
}

void HashMap::Insert(const Tensor& input_keys,
                     const Tensor& input_values,
                     Tensor& output_buf_indices,
                     Tensor& output_masks) {
    Insert(input_keys, std::vector<Tensor>{input_values}, output_buf_indices,
           output_masks);
}

void HashMap::Insert(const Tensor& input_keys,
                     const std::vector<Tensor>& input_values_soa,
                     Tensor& output_buf_indices,
                     Tensor& output_masks) {
    int64_t length = input_keys.GetLength();
    int64_t new_size = Size() + length;
    int64_t capacity = GetCapacity();

    if (new_size > capacity) {
        Reserve(std::max(new_size, capacity * 2));
    }
    InsertImpl(input_keys, input_values_soa, output_buf_indices, output_masks);
}

void HashMap::Activate(const Tensor& input_keys,
                       Tensor& output_buf_indices,
                       Tensor& output_masks) {
    int64_t length = input_keys.GetLength();
    int64_t new_size = Size() + length;
    int64_t capacity = GetCapacity();

    if (new_size > capacity) {
        Reserve(std::max(new_size, capacity * 2));
    }

    std::vector<Tensor> null_tensors_soa;
    InsertImpl(input_keys, null_tensors_soa, output_buf_indices, output_masks,
               /* is_activate_op */ true);
}

void HashMap::Find(const Tensor& input_keys,
                   Tensor& output_buf_indices,
                   Tensor& output_masks) {
    CheckKeyLength(input_keys);
    CheckKeyCompatibility(input_keys);

    int64_t length = input_keys.GetLength();
    PrepareIndicesOutput(output_buf_indices, length);
    PrepareMasksOutput(output_masks, length);

    device_hashmap_->Find(
            input_keys.GetDataPtr(),
            static_cast<buf_index_t*>(output_buf_indices.GetDataPtr()),
            output_masks.GetDataPtr<bool>(), length);
}

void HashMap::Erase(const Tensor& input_keys, Tensor& output_masks) {
    CheckKeyLength(input_keys);
    CheckKeyCompatibility(input_keys);

    int64_t length = input_keys.GetLength();
    PrepareMasksOutput(output_masks, length);

    device_hashmap_->Erase(input_keys.GetDataPtr(),
                           output_masks.GetDataPtr<bool>(), length);
}

void HashMap::GetActiveIndices(Tensor& output_buf_indices) const {
    int64_t length = device_hashmap_->Size();
    PrepareIndicesOutput(output_buf_indices, length);

    device_hashmap_->GetActiveIndices(
            static_cast<buf_index_t*>(output_buf_indices.GetDataPtr()));
}

void HashMap::Clear() { device_hashmap_->Clear(); }

void HashMap::Save(const std::string& file_name) {
    t::io::WriteHashMap(file_name, *this);
}

HashMap HashMap::Load(const std::string& file_name) {
    return t::io::ReadHashMap(file_name);
}

HashMap HashMap::Clone() const { return To(GetDevice(), /*copy=*/true); }

HashMap HashMap::To(const Device& device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }

    Tensor keys = GetKeyTensor();
    std::vector<Tensor> values = GetValueTensors();

    Tensor active_buf_indices_i32;
    GetActiveIndices(active_buf_indices_i32);
    Tensor active_indices = active_buf_indices_i32.To(core::Int64);

    Tensor active_keys = keys.IndexGet({active_indices}).To(device);
    std::vector<Tensor> soa_active_values;
    for (const auto& value : values) {
        soa_active_values.push_back(
                value.IndexGet({active_indices}).To(device));
    }

    HashMap new_hashmap(GetCapacity(), key_dtype_, key_element_shape_,
                        dtypes_value_, element_shapes_value_, device);
    Tensor buf_indices, masks;
    new_hashmap.Insert(active_keys, soa_active_values, buf_indices, masks);

    return new_hashmap;
}

int64_t HashMap::Size() const { return device_hashmap_->Size(); }

int64_t HashMap::GetCapacity() const { return device_hashmap_->GetCapacity(); }

int64_t HashMap::GetBucketCount() const {
    return device_hashmap_->GetBucketCount();
}

Device HashMap::GetDevice() const { return device_hashmap_->GetDevice(); }

Tensor HashMap::GetKeyTensor() const {
    int64_t capacity = GetCapacity();
    SizeVector key_shape = key_element_shape_;
    key_shape.insert(key_shape.begin(), capacity);
    return Tensor(key_shape, shape_util::DefaultStrides(key_shape),
                  device_hashmap_->GetKeyBuffer().GetDataPtr(), key_dtype_,
                  device_hashmap_->GetKeyBuffer().GetBlob());
}

std::vector<Tensor> HashMap::GetValueTensors() const {
    int64_t capacity = GetCapacity();

    std::vector<Tensor> value_buffers = device_hashmap_->GetValueBuffers();

    std::vector<Tensor> soa_value_tensor;
    for (size_t i = 0; i < element_shapes_value_.size(); ++i) {
        SizeVector value_shape = element_shapes_value_[i];
        value_shape.insert(value_shape.begin(), capacity);

        Dtype value_dtype = dtypes_value_[i];
        soa_value_tensor.push_back(
                Tensor(value_shape, shape_util::DefaultStrides(value_shape),
                       value_buffers[i].GetDataPtr(), value_dtype,
                       value_buffers[i].GetBlob()));
    }
    return soa_value_tensor;
}

Tensor HashMap::GetValueTensor(size_t i) const {
    int64_t capacity = GetCapacity();

    if (i >= dtypes_value_.size()) {
        utility::LogError("Value index ({}) out of bound (>= {})", i,
                          dtypes_value_.size());
    }

    Tensor value_buffer = device_hashmap_->GetValueBuffer(i);

    SizeVector value_shape = element_shapes_value_[i];
    value_shape.insert(value_shape.begin(), capacity);

    Dtype value_dtype = dtypes_value_[i];
    return Tensor(value_shape, shape_util::DefaultStrides(value_shape),
                  value_buffer.GetDataPtr(), value_dtype,
                  value_buffer.GetBlob());
}

std::vector<int64_t> HashMap::BucketSizes() const {
    return device_hashmap_->BucketSizes();
};

float HashMap::LoadFactor() const { return device_hashmap_->LoadFactor(); }

void HashMap::Init(int64_t init_capacity,
                   const Device& device,
                   const HashBackendType& backend) {
    // Key check
    if (key_dtype_.GetDtypeCode() == Dtype::DtypeCode::Undefined) {
        utility::LogError("Undefined key dtype is not allowed.");
    }
    if (key_element_shape_.NumElements() == 0) {
        utility::LogError(
                "Key element shape must contain at least 1 element, "
                "but got 0.");
    }

    // Value check
    if (dtypes_value_.size() != element_shapes_value_.size()) {
        utility::LogError(
                "Size of value_dtype ({}) mismatches with size of "
                "element_shapes_value ({}).",
                dtypes_value_.size(), element_shapes_value_.size());
    }
    for (const auto& value_dtype : dtypes_value_) {
        if (value_dtype.GetDtypeCode() == Dtype::DtypeCode::Undefined) {
            utility::LogError("Undefined value dtype is not allowed.");
        }
    }
    for (const auto& value_element_shape : element_shapes_value_) {
        if (value_element_shape.NumElements() == 0) {
            utility::LogError(
                    "Value element shape must contain at least 1 "
                    "element, but got 0.");
        }
    }

    device_hashmap_ = CreateDeviceHashBackend(
            init_capacity, key_dtype_, key_element_shape_, dtypes_value_,
            element_shapes_value_, device, backend);
}

void HashMap::CheckKeyLength(const Tensor& input_keys) const {
    int64_t key_len = input_keys.GetLength();
    if (key_len == 0) {
        utility::LogError("Input number of keys should > 0, but got 0.");
    }
}

void HashMap::CheckKeyValueLengthCompatibility(
        const Tensor& input_keys,
        const std::vector<Tensor>& input_values_soa) const {
    int64_t key_len = input_keys.GetLength();
    if (key_len == 0) {
        utility::LogError("Input number of keys should > 0, but got 0.");
    }
    for (size_t i = 0; i < input_values_soa.size(); ++i) {
        Tensor input_value = input_values_soa[i];
        if (input_value.GetLength() != key_len) {
            utility::LogError(
                    "Input number of values at {} mismatch with number of "
                    "keys "
                    "{}",
                    key_len, input_value.GetLength());
        }
    }
}

void HashMap::CheckKeyCompatibility(const Tensor& input_keys) const {
    SizeVector input_key_elem_shape(input_keys.GetShape());
    input_key_elem_shape.erase(input_key_elem_shape.begin());

    int64_t input_key_elem_bytesize = input_key_elem_shape.NumElements() *
                                      input_keys.GetDtype().ByteSize();
    int64_t stored_key_elem_bytesize =
            key_element_shape_.NumElements() * key_dtype_.ByteSize();
    if (input_key_elem_bytesize != stored_key_elem_bytesize) {
        utility::LogError(
                "Input key element bytesize ({}) mismatch with stored ({})",
                input_key_elem_bytesize, stored_key_elem_bytesize);
    }
}

void HashMap::CheckValueCompatibility(
        const std::vector<Tensor>& input_values_soa) const {
    if (input_values_soa.size() != element_shapes_value_.size()) {
        utility::LogError(
                "Input number of value arrays ({}) mismatches with stored "
                "({})",
                input_values_soa.size(), element_shapes_value_.size());
    }

    for (size_t i = 0; i < input_values_soa.size(); ++i) {
        Tensor input_value = input_values_soa[i];
        SizeVector input_value_i_elem_shape(input_value.GetShape());
        input_value_i_elem_shape.erase(input_value_i_elem_shape.begin());

        int64_t input_value_i_elem_bytesize =
                input_value_i_elem_shape.NumElements() *
                input_value.GetDtype().ByteSize();

        int64_t stored_value_i_elem_bytesize =
                element_shapes_value_[i].NumElements() *
                dtypes_value_[i].ByteSize();
        if (input_value_i_elem_bytesize != stored_value_i_elem_bytesize) {
            utility::LogError(
                    "Input value[{}] element bytesize ({}) mismatch with "
                    "stored ({})",
                    i, input_value_i_elem_bytesize,
                    stored_value_i_elem_bytesize);
        }
    }
}

void HashMap::PrepareIndicesOutput(Tensor& output_buf_indices,
                                   int64_t length) const {
    if (output_buf_indices.GetLength() != length ||
        output_buf_indices.GetDtype() != core::Int32 ||
        output_buf_indices.GetDevice() != GetDevice()) {
        output_buf_indices = Tensor({length}, core::Int32, GetDevice());
    }
}

void HashMap::PrepareMasksOutput(Tensor& output_masks, int64_t length) const {
    if (output_masks.GetLength() != length ||
        output_masks.GetDtype() != core::Bool ||
        output_masks.GetDevice() != GetDevice()) {
        output_masks = Tensor({length}, core::Bool, GetDevice());
    }
}

}  // namespace core
}  // namespace open3d
