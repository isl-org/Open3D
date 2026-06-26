// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/IndexGetSet.h"

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
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(src.GetDevice());
    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        queue.parallel_for(ai.NumWorkloads(), [ai,
                                               object_byte_size](int64_t idx) {
                 char* dst_bytes = ai.GetOutputPtr(idx);
                 const char* src_bytes = ai.GetInputPtr(idx);
                 for (int64_t i = 0; i < object_byte_size; ++i) {
                     dst_bytes[i] = src_bytes[i];
                 }
             }).wait_and_throw();
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            queue.parallel_for(ai.NumWorkloads(), [ai](int64_t idx) {
                     // char* -> scalar_t* needs reinterpret_cast
                     *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(idx)) =
                             *reinterpret_cast<const scalar_t*>(
                                     ai.GetInputPtr(idx));
                 }).wait_and_throw();
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
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(src.GetDevice());
    if (dtype.IsObject()) {
        int64_t object_byte_size = dtype.ByteSize();
        queue.parallel_for(ai.NumWorkloads(), [ai,
                                               object_byte_size](int64_t idx) {
                 char* dst_bytes = ai.GetOutputPtr(idx);
                 const char* src_bytes = ai.GetInputPtr(idx);
                 for (int64_t i = 0; i < object_byte_size; ++i) {
                     dst_bytes[i] = src_bytes[i];
                 }
             }).wait_and_throw();
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            queue.parallel_for(ai.NumWorkloads(), [ai](int64_t idx) {
                     // char* -> scalar_t* needs reinterpret_cast
                     *reinterpret_cast<scalar_t*>(ai.GetOutputPtr(idx)) =
                             *reinterpret_cast<const scalar_t*>(
                                     ai.GetInputPtr(idx));
                 }).wait_and_throw();
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
