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

#include "Open3D/Core/Kernel/BinaryEW.h"

#include <vector>

#include "Open3D/Core/Broadcast.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              Tensor& dst,
              BinaryEWOpCode op_code) {
    for (auto device :
         std::vector<Device>({rhs.GetDevice(), dst.GetDevice()})) {
        if (lhs.GetDevice() != device) {
            utility::LogError("Device mismatch {} != {}.",
                              lhs.GetDevice().ToString(), device.ToString());
        }
    }
    Device::DeviceType device_type = lhs.GetDevice().GetType();
    if (device_type == Device::DeviceType::CPU) {
        BinaryEWCPU(lhs, rhs, dst, op_code);
    } else if (device_type == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        BinaryEWCUDA(lhs, rhs, dst, op_code);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("BinaryEW: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace open3d
