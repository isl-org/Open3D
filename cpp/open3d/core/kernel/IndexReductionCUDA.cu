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
void LaunchIndexReductionKernel(int64_t dim,
                                const Device& device,
                                const Tensor& index,
                                const Tensor& src,
                                Tensor& dst,
                                const func_t& element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    // index: [N,], src: [N, D], dst: [M, D]
    core::Tensor reshaped_index = index;
    if (src.NumDims() > 1) {
        reshaped_index = reshaped_index.View({-1, 1});
    }
    Indexer indexer({reshaped_index.View({-1}), src.View({-1})}, dst.View({-1}),
                    DtypePolicy::NONE);
    utility::LogInfo("dst ptr = {}", dst.GetDataPtr());

    int64_t broadcasting_elems = 1;
    for (int64_t d = 1; d < src.NumDims(); ++d) {
        broadcasting_elems *= src.GetShape(d);
    }

    std::cout << "Master shape: \n";
    for (int i = 0; i < indexer.NumDims(); ++i) {
        std::cout << indexer.GetMasterShape()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Master stride: \n";
    for (int i = 0; i < indexer.NumDims(); ++i) {
        std::cout << indexer.GetMasterStrides()[i] << " ";
    }
    std::cout << "\n";

    // TensorRef index_tr = indexer.GetInput(0);
    // std::cout << "index_tr shape: \n";
    // for (int i = 0; i < index_tr.ndims_; ++i) {
    //     std::cout << index_tr.shape_[i] << " ";
    // }
    // std::cout << "\n";

    // TensorRef src_tr = indexer.GetInput(1);
    // std::cout << "src_tr shape: \n";
    // for (int i = 0; i < src_tr.ndims_; ++i) {
    //     std::cout << src_tr.shape_[i] << " ";
    // }
    // std::cout << "\n";

    // TensorRef dst_tr = indexer.GetOutput(0);
    // std::cout << "dst_tr shape: \n";
    // for (int i = 0; i < dst_tr.ndims_; ++i) {
    //     std::cout << dst_tr.shape_[i] << " ";
    // }
    // std::cout << "\n";

    utility::LogInfo("broadcasting_elems: {}, index length: {}",
                     broadcasting_elems, reshaped_index.GetLength());

    auto element_func = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
        int reduction_idx = workload_idx / broadcasting_elems;
        int broadcasting_idx = workload_idx % broadcasting_elems;

        const int64_t idx = *(indexer.GetInputPtr<int64_t>(0, reduction_idx));
        int64_t output_idx = idx * broadcasting_elems + broadcasting_idx;
        // printf("output_idx: %ld, output_ptr %p\n", output_idx,
        //        indexer.GetOutputPtr(output_idx));
        // printf("workload idx: %ld reduction idx: %d broadcasting idx: %d idx:
        // "
        //        "%ld output_idx: %ld\n",
        //        workload_idx, reduction_idx, broadcasting_idx, idx,
        //        output_idx);

        element_kernel(indexer.GetInputPtr(1, workload_idx),
                       indexer.GetOutputPtr(output_idx));
    };

    ParallelFor(device, reshaped_index.GetLength() * broadcasting_elems,
                element_func);
    OPEN3D_GET_LAST_CUDA_ERROR("LaunchIndexReductionKernel failed.");
    utility::LogInfo("LaunchIndexReductionKernel done, dst = {}",
                     dst.ToString());
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDASumKernel(const void* src, void* dst) {
    scalar_t* dst_s_ptr = static_cast<scalar_t*>(dst);
    const scalar_t* src_s_ptr = static_cast<const scalar_t*>(src);
    atomicAdd(dst_s_ptr, *src_s_ptr);
}

void IndexAddCUDA_(int64_t dim,
                   const Tensor& index,
                   const Tensor& src,
                   Tensor& dst) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        LaunchIndexReductionKernel(
                dim, src.GetDevice(), index, src, dst,
                [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                    CUDASumKernel<scalar_t>(src, dst);
                });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
