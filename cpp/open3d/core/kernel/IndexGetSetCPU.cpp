// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/IndexGetSet.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename func_t>
static void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                        const func_t& func) {
    ParallelFor(Device("CPU:0"), indexer.NumWorkloads(),
                [&indexer, &func](int64_t i) {
                    func(indexer.GetInputPtr(i), indexer.GetOutputPtr(i));
                });
}

template <typename scalar_t>
static void CPUCopyElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(src);
}

static void CPUCopyObjectElementKernel(const void* src,
                                       void* dst,
                                       int64_t object_byte_size) {
    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);
    memcpy(dst_bytes, src_bytes, object_byte_size);
}

void IndexGetCPU(const Tensor& src,
                 Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const SizeVector& indexed_shape,
                 const SizeVector& indexed_strides) {
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::GET);
    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        LaunchAdvancedIndexerKernel(ai, [&](const void* src, void* dst) {
            CPUCopyObjectElementKernel(src, dst, object_byte_size);
        });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            LaunchAdvancedIndexerKernel(ai, CPUCopyElementKernel<scalar_t>);
        });
    }
}

void IndexSetCPU(const Tensor& src,
                 Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const SizeVector& indexed_shape,
                 const SizeVector& indexed_strides) {
    Dtype dtype = src.GetDtype();
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::SET);
    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        LaunchAdvancedIndexerKernel(ai, [&](const void* src, void* dst) {
            CPUCopyObjectElementKernel(src, dst, object_byte_size);
        });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            LaunchAdvancedIndexerKernel(ai, CPUCopyElementKernel<scalar_t>);
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
