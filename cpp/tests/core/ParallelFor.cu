// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Eigen still applies __host__ __device__ annotation to defaulted functions for
// some classes. This causes compiler errors on Windows + CUDA. Disable CUDA
// support for Eigen until this issue has been fixed upstream.
#define EIGEN_NO_CUDA

#include <vector>

#include "open3d/Macro.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

// CUDA does not allow using extended lambdas in private class scope like in
// googletest's internal test class.
void RunParallelForOn(core::Tensor& tensor) {
    int64_t* tensor_data = tensor.GetDataPtr<int64_t>();
    core::ParallelFor(
            tensor.GetDevice(), tensor.NumElements(),
            [=] OPEN3D_HOST_DEVICE(int64_t idx) { tensor_data[idx] = idx; });
}

TEST(ParallelFor, LambdaCUDA) {
    const core::Device device("CUDA:0");
    const size_t N = 10000000;
    core::Tensor tensor({N, 1}, core::Int64, device);

    RunParallelForOn(tensor);

    core::Tensor tensor_cpu = tensor.To(core::Device("CPU:0"));
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
        ASSERT_EQ(tensor_cpu.GetDataPtr<int64_t>()[i], i);
    }
}

}  // namespace tests
}  // namespace open3d
