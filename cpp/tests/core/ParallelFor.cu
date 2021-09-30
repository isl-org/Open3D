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
