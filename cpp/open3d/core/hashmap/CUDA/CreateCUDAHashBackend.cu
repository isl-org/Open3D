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

#include "open3d/core/hashmap/CUDA/SlabHashBackend.h"
#include "open3d/core/hashmap/CUDA/StdGPUHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/HashMap.h"

namespace open3d {
namespace core {

/// Non-templated factory.
std::shared_ptr<DeviceHashBackend> CreateCUDAHashBackend(
        int64_t init_capacity,
        const Dtype& key_dtype,
        const SizeVector& key_element_shape,
        const std::vector<Dtype>& value_dtypes,
        const std::vector<SizeVector>& value_element_shapes,
        const Device& device,
        const HashBackendType& backend) {
    if (backend != HashBackendType::Default &&
        backend != HashBackendType::Slab &&
        backend != HashBackendType::StdGPU) {
        utility::LogError("Unsupported backend for CUDA hashmap.");
    }

    int64_t dim = key_element_shape.NumElements();

    int64_t key_dsize = dim * key_dtype.ByteSize();

    // TODO: size check
    std::vector<int64_t> value_dsizes;
    for (size_t i = 0; i < value_dtypes.size(); ++i) {
        int64_t dsize_value = value_element_shapes[i].NumElements() *
                              value_dtypes[i].ByteSize();
        value_dsizes.push_back(dsize_value);
    }

    std::shared_ptr<DeviceHashBackend> device_hashmap_ptr;
    if (backend == HashBackendType::Default ||
        backend == HashBackendType::StdGPU) {
        DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(key_dtype, dim, [&] {
            device_hashmap_ptr =
                    std::make_shared<StdGPUHashBackend<key_t, hash_t, eq_t>>(
                            init_capacity, key_dsize, value_dsizes, device);
        });
    } else {  // if (backend == HashBackendType::Slab) {
        DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(key_dtype, dim, [&] {
            device_hashmap_ptr =
                    std::make_shared<SlabHashBackend<key_t, hash_t, eq_t>>(
                            init_capacity, key_dsize, value_dsizes, device);
        });
    }
    return device_hashmap_ptr;
}

}  // namespace core
}  // namespace open3d
