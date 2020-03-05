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

#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Kernel/CPULauncher.h"
#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
static void CPUAddElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) +
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUSubElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) -
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUMulElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) *
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUDivElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) /
                                   *static_cast<const scalar_t*>(rhs);
}

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code) {
    Dtype dtype = dst.GetDtype();
    Indexer indexer({lhs, rhs}, dst, DtypePolicy::ASSERT_SAME);

    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        switch (op_code) {
            case BinaryEWOpCode::Add:
                CPULauncher::LaunchBinaryEWKernel(
                        indexer, CPUAddElementKernel<scalar_t>);
                break;
            case BinaryEWOpCode::Sub:
                CPULauncher::LaunchBinaryEWKernel(
                        indexer, CPUSubElementKernel<scalar_t>);
                break;
            case BinaryEWOpCode::Mul:
                CPULauncher::LaunchBinaryEWKernel(
                        indexer, CPUMulElementKernel<scalar_t>);
                break;
            case BinaryEWOpCode::Div:
                CPULauncher::LaunchBinaryEWKernel(
                        indexer, CPUDivElementKernel<scalar_t>);
                break;
            default:
                break;
        }
    });
}

}  // namespace kernel
}  // namespace open3d
