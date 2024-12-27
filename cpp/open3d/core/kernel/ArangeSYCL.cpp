// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Arange.h"

namespace open3d {
namespace core {
namespace kernel {

void ArangeSYCL(const Tensor& start,
                const Tensor& stop,
                const Tensor& step,
                Tensor& dst) {
    Dtype dtype = start.GetDtype();
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(start.GetDevice());
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
        int64_t n = dst.GetLength();
        queue.parallel_for(n, [=](int64_t i) {
                 dst_ptr[i] = sstart + static_cast<scalar_t>(sstep * i);
             }).wait_and_throw();
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
