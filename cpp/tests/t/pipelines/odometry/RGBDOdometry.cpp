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

#include <numeric>
#include <sstream>
#include <unordered_map>

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/Solve.h"
#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

void SolveNew(const core::Tensor &A, const core::Tensor &B, core::Tensor &X) {
    const core::Device device = A.GetDevice();
    const core::Dtype dtype = A.GetDtype();

    // A and B are modified in-place
    core::Tensor A_copy = A.T().Clone();
    void *A_data = A_copy.GetDataPtr();

    X = B.T().Clone();
    void *B_data = X.GetDataPtr();

    int64_t n = A.GetShape()[0];
    int64_t k = B.GetShape().size() == 2 ? B.GetShape()[1] : 1;

    core::Dtype ipiv_dtype;
    if (sizeof(OPEN3D_CPU_LINALG_INT) == 4) {
        ipiv_dtype = core::Int32;
    } else if (sizeof(OPEN3D_CPU_LINALG_INT) == 8) {
        ipiv_dtype = core::Int64;
    } else {
        utility::LogError("Unsupported OPEN3D_CPU_LINALG_INT type.");
    }
    core::Tensor ipiv = core::Tensor::Empty({n}, ipiv_dtype, device);
    void *ipiv_data = ipiv.GetDataPtr();

    core::SolveCPU(A_data, B_data, ipiv_data, n, k, dtype, device);
    X = X.T();
}

TEST(OdometryPermuteDevices, TestSolve) {
    core::Tensor lhs =
            core::Tensor::Load("/home/yixing/repo/Open3D/debug/AtA.npy");
    core::Tensor rhs =
            core::Tensor::Load("/home/yixing/repo/Open3D/debug/Atb_neg.npy");

    utility::LogInfo("########## To run solve()");
    core::Tensor output;
    SolveNew(lhs, rhs, output);
    utility::LogInfo("########## Solve() done");
}

}  // namespace tests
}  // namespace open3d
