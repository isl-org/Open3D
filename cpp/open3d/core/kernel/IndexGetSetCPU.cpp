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
