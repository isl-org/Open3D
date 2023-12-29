// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Arange.h"

namespace open3d {
namespace core {
namespace kernel {

void ArangeCUDA(const Tensor& start,
                const Tensor& stop,
                const Tensor& step,
                Tensor& dst) {
    CUDAScopedDevice scoped_device(start.GetDevice());
    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
        int64_t n = dst.GetLength();
        ParallelFor(start.GetDevice(), n,
                    [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
                        dst_ptr[workload_idx] =
                                sstart +
                                static_cast<scalar_t>(sstep * workload_idx);
                    });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
