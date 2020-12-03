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

#include "open3d/core/EigenConverter.h"

#include "open3d/core/kernel/CPULauncher.h"

namespace open3d {
namespace core {
namespace eigen_converter {

Eigen::Vector3d TensorToEigenVector3d(const core::Tensor &tensor) {
    // TODO: Tensor::To(dtype, device).
    if (tensor.GetShape() != SizeVector{3}) {
        utility::LogError("Tensor shape must be {3}, but got {}.",
                          tensor.GetShape().ToString());
    }
    core::Tensor dtensor =
            tensor.To(core::Dtype::Float64).Copy(core::Device("CPU:0"));
    return Eigen::Vector3d(dtensor[0].Item<double>(), dtensor[1].Item<double>(),
                           dtensor[2].Item<double>());
}

core::Tensor EigenVector3dToTensor(const Eigen::Vector3d &value,
                                   core::Dtype dtype,
                                   const core::Device &device) {
    // The memory will be copied.
    return core::Tensor(value.data(), {3}, core::Dtype::Float64, device)
            .To(dtype);
}

core::Tensor EigenVector3dVectorToTensor(
        const std::vector<Eigen::Vector3d> &values,
        core::Dtype dtype,
        const core::Device &device) {
    // Init CPU Tensor.
    int64_t num_values = static_cast<int64_t>(values.size());
    core::Tensor tensor_cpu =
            core::Tensor::Empty({num_values, 3}, dtype, Device("CPU:0"));

    // Fill Tensor.
    core::Indexer indexer({tensor_cpu}, tensor_cpu,
                          core::DtypePolicy::ALL_SAME);
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        core::kernel::CPULauncher::LaunchIndexFillKernel(
                indexer, [&](void *ptr, int64_t workload_idx) {
                    // Fills the flattened tensor tensor_cpu[:] with dtype
                    // casting. tensor_cpu[:][i] corresponds to the (i/3)-th
                    // element's (i%3)-th coordinate value.
                    *static_cast<scalar_t *>(ptr) = static_cast<scalar_t>(
                            values[workload_idx / 3](workload_idx % 3));
                });
    });

    // Copy Tensor to device if necessary.
    if (device.GetType() == core::Device::DeviceType::CPU) {
        return tensor_cpu;
    } else {
        return tensor_cpu.Copy(device);
    }
}

}  // namespace eigen_converter
}  // namespace core
}  // namespace open3d
