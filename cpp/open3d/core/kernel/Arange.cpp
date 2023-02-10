// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/Arange.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace core {
namespace kernel {

Tensor Arange(const Tensor& start, const Tensor& stop, const Tensor& step) {
    AssertTensorShape(start, {});
    AssertTensorShape(stop, {});
    AssertTensorShape(step, {});

    const Device device = start.GetDevice();
    AssertTensorDevice(stop, device);
    AssertTensorDevice(step, device);

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

    if (device.IsCPU()) {
        ArangeCPU(start, stop, step, dst);
    } else if (device.IsCUDA()) {
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
