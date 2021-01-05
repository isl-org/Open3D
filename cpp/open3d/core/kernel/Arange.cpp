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

#include "open3d/core/kernel/Arange.h"

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace kernel {

Tensor Arange(const Tensor& start, const Tensor& stop, const Tensor& step) {
    start.AssertShape({}, "Start tensor must have shape {}.");
    stop.AssertShape({}, "Stop tensor must have shape {}.");
    step.AssertShape({}, "Step tensor must have shape {}.");

    Device device = start.GetDevice();
    Device::DeviceType device_type = device.GetType();
    stop.AssertDevice(device,
                      "Stop must have the same dtype and device as start.");
    step.AssertDevice(device,
                      "Step must have the same dtype and device as start.");

    int64_t num_elements = 0;
    bool is_arange_valid = true;

    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstop = stop.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();

        if (sstep == 0) {
            utility::LogError("Step cannot be 0");
        }
        if (sstart == sstop) {
            is_arange_valid = false;
        }

        num_elements = static_cast<int64_t>(
                std::ceil(static_cast<double>(sstop - sstart) /
                          static_cast<double>(sstep)));
        if (num_elements <= 0) {
            is_arange_valid = false;
        }
    });

    // Special case.
    if (!is_arange_valid) {
        return Tensor({0}, dtype, device);
    }

    // Input parameters.
    std::unordered_map<std::string, core::Tensor> srcs = {
            {"start", start},
            {"step", step},
    };

    // Output.
    Tensor dst = Tensor({num_elements}, dtype, device);

    if (device_type == Device::DeviceType::CPU) {
        ArangeCPU(start, stop, step, dst);
    } else if (device_type == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ArangeCUDA(start, stop, step, dst);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Arange: Unimplemented device.");
    }

    return dst;
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
