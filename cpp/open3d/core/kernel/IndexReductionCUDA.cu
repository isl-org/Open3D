// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include <type_traits>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename func_t>
void LaunchIndexReductionKernel(const Device& device,
                                const Tensor& index,
                                const Tensor& src,
                                Tensor& dst,
                                const func_t& element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
    Indexer indexer({index, src}, dst, DtypePolicy::NONE);
    auto element_func = [=] OPEN3D_HOST_DEVICE(int64_t i) {
        const int64_t idx = *(indexer.GetInputPtr<int64_t>(0, i));
        element_kernel(indexer.GetInputPtr(1, i), indexer.GetOutputPtr(idx));
    };
    utility::LogInfo(
            "LaunchIndexReductionKernel: NumWorkloads = {}, index shape: {}, "
            "src shape: {}, dst shape: {}",
            index.GetLength(), index.GetShape(), src.GetShape(),
            dst.GetShape());

    ParallelFor(device, index.GetLength(), element_func);
    OPEN3D_GET_LAST_CUDA_ERROR("LaunchIndexReductionKernel failed.");
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDASumKernel(const void* src, void* dst) {
    scalar_t* dst_s_ptr = static_cast<scalar_t*>(dst);
    const scalar_t* src_s_ptr = static_cast<const scalar_t*>(src);
    atomicAdd(dst_s_ptr, *src_s_ptr);
}

void IndexSumCUDA_(const Tensor& index, const Tensor& src, Tensor& dst) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        LaunchIndexReductionKernel(
                src.GetDevice(), index, src, dst,
                [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                    CUDASumKernel<scalar_t>(src, dst);
                });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
