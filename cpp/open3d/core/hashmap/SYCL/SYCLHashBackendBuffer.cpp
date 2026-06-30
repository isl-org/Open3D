// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <sycl/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"

namespace open3d {
namespace core {

// Initialize the index heap to the identity permutation [0, 1, ..., N-1] so
// that every buffer slot is initially free. Mirrors CPUResetHeap/CUDAResetHeap.
void SYCLResetHeap(Tensor &heap) {
    uint32_t *heap_ptr = heap.GetDataPtr<uint32_t>();
    const int64_t capacity = heap.GetLength();
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(heap.GetDevice());
    queue.parallel_for(capacity, [=](int64_t i) {
             heap_ptr[i] = static_cast<uint32_t>(i);
         }).wait_and_throw();
}

}  // namespace core
}  // namespace open3d
