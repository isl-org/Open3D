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

#include "open3d/core/hashmap/DeviceHashmap.h"

#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {

std::shared_ptr<DeviceHashmap> CreateDeviceHashmap(
        int64_t init_capacity,
        const Dtype& dtype_key,
        const Dtype& dtype_value,
        const SizeVector& element_shape_key,
        const SizeVector& element_shape_value,
        const Device& device,
        const HashmapBackend& backend) {
    if (device.GetType() == Device::DeviceType::CPU) {
        return CreateCPUHashmap(init_capacity, dtype_key, dtype_value,
                                element_shape_key, element_shape_value, device,
                                backend);
    }
#if defined(BUILD_CUDA_MODULE)
    else if (device.GetType() == Device::DeviceType::CUDA) {
        return CreateCUDAHashmap(init_capacity, dtype_key, dtype_value,
                                 element_shape_key, element_shape_value, device,
                                 backend);
    }
#endif
    else {
        utility::LogError("[CreateDeviceHashmap]: Unimplemented device");
    }
}

}  // namespace core
}  // namespace open3d
