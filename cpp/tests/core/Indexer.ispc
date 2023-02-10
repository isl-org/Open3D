// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Indexer.isph"
#include "open3d/core/ParallelFor.isph"

static inline void GetInputPointers(int64 idx,
                                    uniform Indexer* uniform self,
                                    uniform int64 input_idx,
                                    void** uniform input_ptrs) {
    input_ptrs[idx] = (void*)Indexer_GetInputPtr(self, input_idx, idx);
}

static inline void GetOutputPointers(int64 idx,
                                     uniform Indexer* uniform self,
                                     void** uniform output_ptrs) {
    output_ptrs[idx] = (void*)Indexer_GetOutputPtr(self, idx);
}

static inline void GetOutputPointers(int64 idx,
                                     uniform Indexer* uniform self,
                                     uniform int64 output_idx,
                                     void** uniform output_ptrs) {
    output_ptrs[idx] = (void*)Indexer_GetOutputPtr(self, output_idx, idx);
}

OPEN3D_EXPORT_VECTORIZED(GetInputPointersKernel,
                         GetInputPointers,
                         uniform Indexer* uniform,
                         uniform int64,
                         void** uniform)

OPEN3D_EXPORT_VECTORIZED(GetOutputPointersKernel_Zero,
                         GetOutputPointers,
                         uniform Indexer* uniform,
                         void** uniform)

OPEN3D_EXPORT_VECTORIZED(GetOutputPointersKernel,
                         GetOutputPointers,
                         uniform Indexer* uniform,
                         uniform int64,
                         void** uniform)
