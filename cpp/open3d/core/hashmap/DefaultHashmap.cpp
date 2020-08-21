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

#include "HashmapBase.h"

#include <unordered_map>

namespace open3d {
namespace core {
std::shared_ptr<DefaultHashmap> CreateDefaultHashmap(size_t init_capacity,
                                                     size_t dsize_key,
                                                     size_t dsize_value,
                                                     Device device) {
    return CreateDefaultHashmap(init_capacity / kDefaultElemsPerBucket,
                                init_capacity, dsize_key, dsize_value, device);
}

std::shared_ptr<DefaultHashmap> CreateDefaultHashmap(size_t init_buckets,
                                                     size_t init_capacity,
                                                     size_t dsize_key,
                                                     size_t dsize_value,
                                                     Device device) {
    static std::unordered_map<Device::DeviceType,
                              std::function<std::shared_ptr<DefaultHashmap>(
                                      size_t, size_t, size_t, size_t, Device)>,
                              utility::hash_enum_class::hash>
            map_device_type_to_hashmap_constructor = {
                {Device::DeviceType::CPU, _factory::CreateDefaultCPUHashmap},
#if defined(BUILD_CUDA_MODULE)
                {Device::DeviceType::CUDA, _factory::CreateDefaultCUDAHashmap}
#endif
            };

    if (map_device_type_to_hashmap_constructor.find(device.GetType()) ==
        map_device_type_to_hashmap_constructor.end()) {
        utility::LogError("CreateDefaultHashmap: Unimplemented device");
    }

    auto constructor =
            map_device_type_to_hashmap_constructor.at(device.GetType());
    return constructor(init_buckets, init_capacity, dsize_key, dsize_value,
                       device);
}
}  // namespace core
}  // namespace open3d
