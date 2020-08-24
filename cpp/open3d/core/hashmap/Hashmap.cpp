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

#include <unordered_map>

#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {
std::shared_ptr<DefaultDeviceHashmap> CreateDefaultDeviceHashmap(
        size_t init_capacity,
        size_t dsize_key,
        size_t dsize_value,
        Device device) {
    return CreateDefaultDeviceHashmap(init_capacity / kDefaultElemsPerBucket,
                                      init_capacity, dsize_key, dsize_value,
                                      device);
}

std::shared_ptr<DefaultDeviceHashmap> CreateDefaultDeviceHashmap(
        size_t init_buckets,
        size_t init_capacity,
        size_t dsize_key,
        size_t dsize_value,
        Device device) {
    static std::unordered_map<
            Device::DeviceType,
            std::function<std::shared_ptr<DefaultDeviceHashmap>(
                    size_t, size_t, size_t, size_t, Device)>,
            utility::hash_enum_class>
            map_device_type_to_hashmap_constructor = {
                {Device::DeviceType::CPU, CreateDefaultCPUHashmap},
#if defined(BUILD_CUDA_MODULE)
                {Device::DeviceType::CUDA, CreateDefaultCUDAHashmap}
#endif
            };

    if (map_device_type_to_hashmap_constructor.find(device.GetType()) ==
        map_device_type_to_hashmap_constructor.end()) {
        utility::LogError("CreateDefaultDeviceHashmap: Unimplemented device");
    }

    auto constructor =
            map_device_type_to_hashmap_constructor.at(device.GetType());
    return constructor(init_buckets, init_capacity, dsize_key, dsize_value,
                       device);
}

Hashmap::Hashmap(size_t init_buckets,
                 size_t init_capacity,
                 size_t dsize_key,
                 size_t dsize_value,
                 Device device) {
    device_hashmap_ = CreateDefaultDeviceHashmap(
            init_buckets, init_capacity, dsize_key, dsize_value, device);
}

Hashmap::Hashmap(size_t init_capacity,
                 size_t dsize_key,
                 size_t dsize_value,
                 Device device) {
    device_hashmap_ = CreateDefaultDeviceHashmap(init_capacity, dsize_key,
                                                 dsize_value, device);
}

void Hashmap::Rehash(size_t buckets) {
    return device_hashmap_->Rehash(buckets);
}

void Hashmap::Insert(const void* input_keys,
                     const void* input_values,
                     iterator_t* output_iterators,
                     bool* output_masks,
                     size_t count) {
    return device_hashmap_->Insert(input_keys, input_values, output_iterators,
                                   output_masks, count);
}

void Hashmap::Activate(const void* input_keys,
                       iterator_t* output_iterators,
                       bool* output_masks,
                       size_t count) {
    return device_hashmap_->Activate(input_keys, output_iterators, output_masks,
                                     count);
}

void Hashmap::Find(const void* input_keys,
                   iterator_t* output_iterators,
                   bool* output_masks,
                   size_t count) {
    return device_hashmap_->Find(input_keys, output_iterators, output_masks,
                                 count);
}

void Hashmap::Erase(const void* input_keys, bool* output_masks, size_t count) {
    return device_hashmap_->Erase(input_keys, output_masks, count);
}

size_t Hashmap::GetIterators(iterator_t* output_iterators) {
    return device_hashmap_->GetIterators(output_iterators);
}

void Hashmap::UnpackIterators(const iterator_t* input_iterators,
                              const bool* input_masks,
                              void* output_keys,
                              void* output_values,
                              size_t count) {
    return device_hashmap_->UnpackIterators(input_iterators, input_masks,
                                            output_keys, output_values, count);
}

void Hashmap::AssignIterators(iterator_t* input_iterators,
                              const bool* input_masks,
                              const void* input_values,
                              size_t count) {
    return device_hashmap_->AssignIterators(input_iterators, input_masks,
                                            input_values, count);
}

size_t Hashmap::Size() { return device_hashmap_->Size(); }

std::vector<size_t> Hashmap::BucketSizes() {
    return device_hashmap_->BucketSizes();
}

float Hashmap::LoadFactor() { return device_hashmap_->LoadFactor(); }

}  // namespace core
}  // namespace open3d
