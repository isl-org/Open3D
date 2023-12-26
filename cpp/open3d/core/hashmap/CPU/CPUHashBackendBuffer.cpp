// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/hashmap/HashBackendBuffer.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
void CPUResetHeap(Tensor& heap) {
    uint32_t* heap_ptr = heap.GetDataPtr<uint32_t>();
    int64_t capacity = heap.GetLength();

#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < capacity; ++i) {
        heap_ptr[i] = i;
    }
};
}  // namespace core
}  // namespace open3d
