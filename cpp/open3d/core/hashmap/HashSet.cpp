// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/hashmap/HashSet.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/t/io/HashMapIO.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

HashSet::HashSet(int64_t init_capacity,
                 const Dtype& key_dtype,
                 const SizeVector& key_element_shape,
                 const Device& device,
                 const HashBackendType& backend) {
    internal_ = std::make_shared<HashMap>(
            init_capacity, key_dtype, key_element_shape, std::vector<Dtype>{},
            std::vector<SizeVector>{}, device, backend);
}

void HashSet::Reserve(int64_t capacity) { return internal_->Reserve(capacity); }

std::pair<Tensor, Tensor> HashSet::Insert(const Tensor& input_keys) {
    Tensor output_buf_indices, output_masks;
    Insert(input_keys, output_buf_indices, output_masks);
    return std::make_pair(output_buf_indices, output_masks);
}

std::pair<Tensor, Tensor> HashSet::Find(const Tensor& input_keys) {
    Tensor output_buf_indices, output_masks;
    Find(input_keys, output_buf_indices, output_masks);
    return std::make_pair(output_buf_indices, output_masks);
}

Tensor HashSet::Erase(const Tensor& input_keys) {
    Tensor output_masks;
    Erase(input_keys, output_masks);
    return output_masks;
}

Tensor HashSet::GetActiveIndices() const {
    Tensor output_buf_indices;
    GetActiveIndices(output_buf_indices);
    return output_buf_indices;
}

void HashSet::Insert(const Tensor& input_keys,
                     Tensor& output_buf_indices,
                     Tensor& output_masks) {
    internal_->Insert(input_keys, std::vector<Tensor>{}, output_buf_indices,
                      output_masks);
}

void HashSet::Find(const Tensor& input_keys,
                   Tensor& output_buf_indices,
                   Tensor& output_masks) {
    internal_->Find(input_keys, output_buf_indices, output_masks);
}

void HashSet::Erase(const Tensor& input_keys, Tensor& output_masks) {
    internal_->Erase(input_keys, output_masks);
}

void HashSet::GetActiveIndices(Tensor& output_buf_indices) const {
    internal_->GetActiveIndices(output_buf_indices);
}

void HashSet::Clear() { internal_->Clear(); }

void HashSet::Save(const std::string& file_name) {
    t::io::WriteHashMap(file_name, *internal_);
}

HashSet HashSet::Load(const std::string& file_name) {
    HashMap internal = t::io::ReadHashMap(file_name);
    return HashSet(internal);
}

HashSet HashSet::Clone() const {
    HashMap internal_cloned = internal_->To(GetDevice(), /*copy=*/true);
    return HashSet(internal_cloned);
}

HashSet HashSet::To(const Device& device, bool copy) const {
    HashMap internal_converted = internal_->To(device, copy);
    return HashSet(internal_converted);
}

int64_t HashSet::Size() const { return internal_->Size(); }

int64_t HashSet::GetCapacity() const { return internal_->GetCapacity(); }

int64_t HashSet::GetBucketCount() const { return internal_->GetBucketCount(); }

Device HashSet::GetDevice() const { return internal_->GetDevice(); }

Tensor HashSet::GetKeyTensor() const { return internal_->GetKeyTensor(); }

std::vector<int64_t> HashSet::BucketSizes() const {
    return internal_->BucketSizes();
};

float HashSet::LoadFactor() const { return internal_->LoadFactor(); }

std::shared_ptr<DeviceHashBackend> HashSet::GetDeviceHashBackend() const {
    return internal_->GetDeviceHashBackend();
}

HashSet::HashSet(const HashMap& internal_hashmap) {
    internal_ = std::make_shared<HashMap>(internal_hashmap);
}

}  // namespace core
}  // namespace open3d
