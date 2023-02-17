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
