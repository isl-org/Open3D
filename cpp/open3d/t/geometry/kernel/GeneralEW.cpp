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

#include "open3d/t/geometry/kernel/GeneralEW.h"

#include <vector>

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

void GeneralEW(const std::unordered_map<std::string, Tensor>& srcs,
               std::unordered_map<std::string, Tensor>& dsts,
               GeneralEWOpCode op_code) {
    // srcs cannot be empty. dsts can be empty on initialization and emplaced at
    // runtime in specific kernels.
    if (srcs.size() == 0) {
        utility::LogError(
                "[GeneralEW]: one or more inputs expected, but received 0.");
    }

    // srcs and dsts must be on the same device.
    Device device = srcs.begin()->second.GetDevice();
    for (auto it = srcs.begin(); it != srcs.end(); ++it) {
        if (device != it->second.GetDevice()) {
            utility::LogError("[GeneralEW]: incompatible device in inputs");
        }
    }
    for (auto it = dsts.begin(); it != dsts.end(); ++it) {
        if (device != it->second.GetDevice()) {
            utility::LogError("[GeneralEW]: incompatible device in outputs");
        }
    }

    // We don't assume shape consistency: general ops are less constrained.
    Device::DeviceType device_type = device.GetType();
    if (device_type == Device::DeviceType::CPU) {
        GeneralEWCPU(srcs, dsts, op_code);
    } else if (device_type == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        GeneralEWCUDA(srcs, dsts, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("GeneralEW: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
