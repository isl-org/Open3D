// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/BlockCopyDispatch.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace kernel {

void IndexGetSYCL(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::GET);
    sycl::queue queue = sy::GetQueue(src.GetDevice());
    if (dtype.IsObject()) {
        const int64_t object_byte_size = dtype.ByteSize();
        const int64_t block_size =
                GetLargestAlignedObjectBlockSize(object_byte_size);
        core::ParallelFor(
                queue, ai.NumWorkloads(),
                [ai, object_byte_size, block_size](int64_t idx) {
                    DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(block_size, [&]() {
                        const int64_t blocks = object_byte_size / block_size;
                        const block_t* src = reinterpret_cast<const block_t*>(
                                ai.GetInputPtr(idx));
                        block_t* dst = reinterpret_cast<block_t*>(
                                ai.GetOutputPtr(idx));
                        for (int64_t b = 0; b < blocks; ++b) {
                            dst[b] = src[b];
                        }
                    });
                });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            core::ParallelFor(queue, ai.NumWorkloads(), [ai](int64_t idx) {
                // char* -> scalar_t* needs reinterpret_cast
                *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(idx)) =
                        *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(idx));
            });
        });
    }
}

void IndexSetSYCL(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::SET);
    sycl::queue queue = sy::GetQueue(src.GetDevice());
    if (dtype.IsObject()) {
        const int64_t object_byte_size = dtype.ByteSize();
        const int64_t block_size =
                GetLargestAlignedObjectBlockSize(object_byte_size);
        core::ParallelFor(
                queue, ai.NumWorkloads(),
                [ai, object_byte_size, block_size](int64_t idx) {
                    DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(block_size, [&]() {
                        const int64_t blocks = object_byte_size / block_size;
                        const block_t* src = reinterpret_cast<const block_t*>(
                                ai.GetInputPtr(idx));
                        block_t* dst = reinterpret_cast<block_t*>(
                                ai.GetOutputPtr(idx));
                        for (int64_t b = 0; b < blocks; ++b) {
                            dst[b] = src[b];
                        }
                    });
                });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            core::ParallelFor(queue, ai.NumWorkloads(), [ai](int64_t idx) {
                // char* -> scalar_t* needs reinterpret_cast
                *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(idx)) =
                        *reinterpret_cast<const scalar_t*>(ai.GetInputPtr(idx));
            });
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
