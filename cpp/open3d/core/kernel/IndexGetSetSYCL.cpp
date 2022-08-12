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

static void RunIndexGetSetSYCL(const AdvancedIndexer& ai,
                               const Device& device) {
    const int64_t object_byte_size = ai.ElementByteSize();
    const int64_t num_workloads = ai.NumWorkloads();

    sy::queue& queue = sycl::GetDefaultQueue(device);
    queue.submit([&](sy::handler& h) {
             h.parallel_for(num_workloads, [ai, object_byte_size](int64_t i) {
                 const char* src_ptr =
                         static_cast<const char*>(ai.GetInputPtr(i));
                 char* dst_ptr = static_cast<char*>(ai.GetOutputPtr(i));
                 for (int64_t j = 0; j < object_byte_size; j++) {
                     dst_ptr[j] = src_ptr[j];
                 }
             });
         }).wait();
}

void IndexGetSYCL(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::GET);
    RunIndexGetSetSYCL(ai, src.GetDevice());
}

void IndexSetSYCL(const Tensor& src,
                  Tensor& dst,
                  const std::vector<Tensor>& index_tensors,
                  const SizeVector& indexed_shape,
                  const SizeVector& indexed_strides) {
    AdvancedIndexer ai(src, dst, index_tensors, indexed_shape, indexed_strides,
                       AdvancedIndexer::AdvancedIndexerMode::SET);
    RunIndexGetSetSYCL(ai, src.GetDevice());
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
