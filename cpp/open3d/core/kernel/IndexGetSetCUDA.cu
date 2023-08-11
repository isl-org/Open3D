// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/IndexGetSet.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename func_t>
void LaunchAdvancedIndexerKernel(const Device& device,
                                 const AdvancedIndexer& indexer,
                                 const func_t& element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
    auto element_func = [=] OPEN3D_HOST_DEVICE(int64_t i) {
        element_kernel(indexer.GetInputPtr(i), indexer.GetOutputPtr(i));
    };
    ParallelFor(device, indexer.NumWorkloads(), element_func);
    OPEN3D_GET_LAST_CUDA_ERROR("LaunchAdvancedIndexerKernel failed.");
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDACopyElementKernel(const void* src,
                                                     void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(src);
}

static OPEN3D_HOST_DEVICE void CUDACopyObjectElementKernel(
        const void* src, void* dst, int64_t object_byte_size) {
    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);
    for (int i = 0; i < object_byte_size; ++i) {
        dst_bytes[i] = src_bytes[i];
    }
}

void IndexGetCUDA(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    CUDAScopedDevice scoped_device(src.GetDevice());
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::GET);

    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        LaunchAdvancedIndexerKernel(
                src.GetDevice(), ai,
                [=] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                    CUDACopyObjectElementKernel(src, dst, object_byte_size);
                });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            LaunchAdvancedIndexerKernel(
                    src.GetDevice(), ai,
                    // Need to wrap as extended CUDA lambda function
                    [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                        CUDACopyElementKernel<scalar_t>(src, dst);
                    });
        });
    }
}

void IndexSetCUDA(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    CUDAScopedDevice scoped_device(src.GetDevice());
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::SET);

    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        LaunchAdvancedIndexerKernel(
                src.GetDevice(), ai,
                [=] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                    CUDACopyObjectElementKernel(src, dst, object_byte_size);
                });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            LaunchAdvancedIndexerKernel(
                    src.GetDevice(), ai,
                    // Need to wrap as extended CUDA lambda function
                    [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                        CUDACopyElementKernel<scalar_t>(src, dst);
                    });
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
