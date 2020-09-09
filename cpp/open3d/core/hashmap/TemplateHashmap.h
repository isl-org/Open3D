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

#pragma once

// Low level hashmap interface for advanced usages who wish to modify hash
// function and/or equal function.
// Users must create a .cu file and compile with
// nvcc to use customized GPU hashmap.
//
// APIs are available in HashmapBase.h.
// Include path: TemplatedHashmap.h -> HashmapCPU.hpp -> HashmapBase.h
//                                |                      ^
//                                |--> HashmapCUDA.cuh --|
//                                        (CUDA code)
//
// .cpp targets only include CPU part that can be compiled by non-nvcc
// compilers even if BUILD_CUDA_MODULE is enabled.
// .cu targets include both.

#include "open3d/core/hashmap/CPU/HashmapCPU.hpp"

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
#include "open3d/core/hashmap/CUDA/HashmapCUDA.cuh"
#endif

#include <unordered_map>

namespace open3d {
namespace core {

template <typename Hash, typename KeyEq>
std::shared_ptr<DeviceHashmap<Hash, KeyEq>> CreateTemplateDeviceHashmap(
        size_t init_buckets,
        size_t init_capacity,
        size_t dsize_key,
        size_t dsize_value,
        const Device &device) {
    static std::unordered_map<
            Device::DeviceType,
            std::function<std::shared_ptr<Hashmap<Hash, KeyEq>>(
                    size_t, size_t, size_t, size_t, Device)>,
            utility::hash_enum_class::hash>
            map_device_type_to_hashmap_constructor =
    { {Device::DeviceType::CPU, CreateTemplateCPUHashmap<Hash, KeyEq>},
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
      {Device::DeviceType::CUDA, CreateTemplateCUDAHashmap<Hash, KeyEq>}
#endif
    };

    if (map_device_type_to_hashmap_constructor.find(device.GetType()) ==
        map_device_type_to_hashmap_constructor.end()) {
        utility::LogError("CreateHashmap: Unimplemented device");
    }

    auto constructor =
            map_device_type_to_hashmap_constructor.at(device.GetType());
    return constructor(init_buckets, init_capacity, dsize_key, dsize_value,
                       device);
}
}  // namespace core
}  // namespace open3d
