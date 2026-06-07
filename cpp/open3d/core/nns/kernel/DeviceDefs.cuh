// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) Facebook, Inc. and its affiliates.
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ----------------------------------------------------------------------------
// original path: faiss/faiss/gpu/utils/DeviceDefs.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

namespace open3d {
namespace core {

// We require at least CUDA 8.0 for compilation
#if CUDA_VERSION < 8000
#error "CUDA >= 8.0 is required"
#endif

// This FAISS-derived warp-select treats a "warp" as a 32-lane bitonic-sort
// group: the queue layout, the shfl_xor butterfly (width=kWarpSize), the
// per-thread/warp-queue register counts, and the smallest instantiated queue
// (NumWarpQ==32, see BlockSelectFloat*.cu) are all built around 32 lanes. On a
// 64-lane CDNA wavefront we therefore run the algorithm as two independent
// 32-lane groups: kWarpSize stays 32, every
// shfl uses width=32 so exchanges stay within a group, and getLaneId() returns
// the 0..31 lane *within* the group (PtxUtils.cuh). This is correct on wave32
// (gfx11xx) and wave64 (gfx9xx) alike; do NOT raise it to 64 -- that collapses
// the NumWarpQ<64 queues (kNumWarpQRegisters = NumWarpQ / kWarpSize == 0).
constexpr int kWarpSize = 32;

// This is a memory barrier for intra-warp writes to shared memory. The select
// kernels are warp-convergent (no warp-granularity divergence), so a
// full-wavefront barrier is safe and covers both 32-lane groups on wave64.
__forceinline__ __device__ void warpFence() {
#if defined(USE_HIP)
    __syncthreads();
#elif CUDA_VERSION >= 9000
    __syncwarp();
#else
    // For the time being, assume synchronicity.
    //  __threadfence_block();
#endif
}

#if CUDA_VERSION > 9000
// Based on the CUDA version (we assume what version of nvcc/ptxas we were
// compiled with), the register allocation algorithm is much better, so only
// enable the 2048 selection code if we are above 9.0 (9.2 seems to be ok)
#define GPU_MAX_SELECTION_K 2048
#else
#define GPU_MAX_SELECTION_K 1024
#endif

}  // namespace core
}  // namespace open3d
