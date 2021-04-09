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

#include "open3d/core/hashmap/CUDA/SlabHashmap.h"
#include "open3d/core/hashmap/CUDA/StdGPUHashmap.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/Hashmap.h"

namespace open3d {
namespace core {

/// Non-templated factory.
std::shared_ptr<DeviceHashmap> CreateCUDAHashmap(
        int64_t init_capacity,
        const Dtype& dtype_key,
        const Dtype& dtype_value,
        const SizeVector& element_shape_key,
        const SizeVector& element_shape_value,
        const Device& device,
        const HashmapBackend& backend) {
    if (backend != HashmapBackend::Default && backend != HashmapBackend::Slab &&
        backend != HashmapBackend::StdGPU) {
        utility::LogError("Unsupported backend for CUDA hashmap.");
    }

    int64_t dim = element_shape_key.NumElements();

    int64_t dsize_key = dim * dtype_key.ByteSize();
    int64_t dsize_value =
            element_shape_value.NumElements() * dtype_value.ByteSize();

    std::shared_ptr<DeviceHashmap> device_hashmap_ptr;
    if (backend == HashmapBackend::Default ||
        backend == HashmapBackend::StdGPU) {
        DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(dtype_key, dim, [&] {
            device_hashmap_ptr = std::make_shared<StdGPUHashmap<key_t, hash_t>>(
                    init_capacity, dsize_key, dsize_value, device);
        });
    } else {  // if (backend == HashmapBackend::Slab) {
        DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(dtype_key, dim, [&] {
            device_hashmap_ptr = std::make_shared<SlabHashmap<key_t, hash_t>>(
                    init_capacity, dsize_key, dsize_value, device);
        });
    }
    return device_hashmap_ptr;
}

}  // namespace core
}  // namespace open3d
