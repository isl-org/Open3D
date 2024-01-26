// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/hashmap/HashBackendBuffer.h"
#include "open3d/utility/Parallel.h"

#include <tbb/parallel_for.h>

namespace open3d {
namespace core {
void CPUResetHeap(Tensor& heap) {
    uint32_t* heap_ptr = heap.GetDataPtr<uint32_t>();
    int64_t capacity = heap.GetLength();

    tbb::parallel_for(tbb::blocked_range<int64_t>(
            0, capacity, utility::DefaultGrainSizeTBB()),
            [&](const tbb::blocked_range<int64_t>& range){
        for (int64_t i = range.begin(); i < range.end(); ++i) {
            heap_ptr[i] = i;
        }
    });
};
}  // namespace core
}  // namespace open3d
