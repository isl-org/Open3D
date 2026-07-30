// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file CreateSYCLHashBackend.cpp
/// \brief Dtype-dispatched factory for \ref SYCLHashBackend.

#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/core/hashmap/SYCL/SYCLHashBackend.h"

namespace open3d {
namespace core {

/// Instantiate \ref SYCLHashBackend for \p key_dtype / element shape and value
/// dtypes. Only \ref HashBackendType::Default is supported on SYCL.
std::shared_ptr<DeviceHashBackend> CreateSYCLHashBackend(
        int64_t init_capacity,
        const Dtype& key_dtype,
        const SizeVector& key_element_shape,
        const std::vector<Dtype>& value_dtypes,
        const std::vector<SizeVector>& value_element_shapes,
        const Device& device,
        const HashBackendType& backend) {
    if (backend != HashBackendType::Default) {
        utility::LogError("Unsupported backend for SYCL hashmap.");
    }

    int64_t dim = key_element_shape.NumElements();
    int64_t key_dsize = dim * key_dtype.ByteSize();

    std::vector<int64_t> value_dsizes;
    for (size_t i = 0; i < value_dtypes.size(); ++i) {
        int64_t dsize_value = value_element_shapes[i].NumElements() *
                              value_dtypes[i].ByteSize();
        value_dsizes.push_back(dsize_value);
    }

    std::shared_ptr<DeviceHashBackend> device_hashmap_ptr;
    DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(key_dtype, dim, [&] {
        device_hashmap_ptr =
                std::make_shared<SYCLHashBackend<key_t, hash_t, eq_t>>(
                        init_capacity, key_dsize, value_dsizes, device);
    });
    return device_hashmap_ptr;
}

}  // namespace core
}  // namespace open3d
