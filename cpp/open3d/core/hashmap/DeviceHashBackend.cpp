// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/hashmap/DeviceHashBackend.h"

#include "open3d/core/hashmap/HashMap.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

std::shared_ptr<DeviceHashBackend> CreateDeviceHashBackend(
        int64_t init_capacity,
        const Dtype& key_dtype,
        const SizeVector& key_element_shape,
        const std::vector<Dtype>& value_dtypes,
        const std::vector<SizeVector>& value_element_shapes,
        const Device& device,
        const HashBackendType& backend) {
    if (device.IsCPU()) {
        return CreateCPUHashBackend(init_capacity, key_dtype, key_element_shape,
                                    value_dtypes, value_element_shapes, device,
                                    backend);
    }
#if defined(BUILD_CUDA_MODULE)
    else if (device.IsCUDA()) {
        return CreateCUDAHashBackend(init_capacity, key_dtype,
                                     key_element_shape, value_dtypes,
                                     value_element_shapes, device, backend);
    }
#endif
    else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace core
}  // namespace open3d
