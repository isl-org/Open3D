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
