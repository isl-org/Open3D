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
// original path: faiss/faiss/gpu/utils/PtxUtils.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

namespace open3d {
namespace core {

#if defined(USE_HIP)

// HIP has no inline PTX; provide the same helpers with HIP/clang builtins.
// getLaneId() returns the lane within the 32-lane bitonic group (0..31), so on
// a 64-lane wavefront the two halves each behave as a 32-lane NVIDIA warp (see
// DeviceDefs.cuh). These select kernels launch 1-D blocks, so the in-group
// lane is threadIdx.x & 31.
#define GET_BITFIELD_U32(OUT, VAL, POS, LEN) \
    OUT = ((VAL) >> (POS)) & ((1u << (LEN)) - 1u)

#define GET_BITFIELD_U64(OUT, VAL, POS, LEN) \
    OUT = ((VAL) >> (POS)) & ((1ull << (LEN)) - 1ull)

__device__ __forceinline__ unsigned int getBitfield(unsigned int val,
                                                     int pos,
                                                     int len) {
    return (val >> pos) & ((1u << len) - 1u);
}

__device__ __forceinline__ uint64_t getBitfield(uint64_t val, int pos, int len) {
    return (val >> pos) & ((1ull << len) - 1ull);
}

__device__ __forceinline__ unsigned int setBitfield(unsigned int val,
                                                     unsigned int toInsert,
                                                     int pos,
                                                     int len) {
    unsigned int mask = ((1u << len) - 1u) << pos;
    return (val & ~mask) | ((toInsert << pos) & mask);
}

__device__ __forceinline__ int getLaneId() {
    return static_cast<int>(threadIdx.x & 31u);
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned lane = threadIdx.x & 31u;
    return (1u << lane) - 1u;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned lane = threadIdx.x & 31u;
    return (lane >= 31u) ? 0xffffffffu : ((1u << (lane + 1u)) - 1u);
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
    return ~getLaneMaskLe();
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
    return ~getLaneMaskLt();
}

__device__ __forceinline__ void namedBarrierWait(int name, int numThreads) {
    __syncthreads();
}

__device__ __forceinline__ void namedBarrierArrived(int name, int numThreads) {
    __syncthreads();
}

#else

// defines to simplify the SASS assembly structure file/line in the profiler
#define GET_BITFIELD_U32(OUT, VAL, POS, LEN) \
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(OUT) : "r"(VAL), "r"(POS), "r"(LEN));

#define GET_BITFIELD_U64(OUT, VAL, POS, LEN) \
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(OUT) : "l"(VAL), "r"(POS), "r"(LEN));

__device__ __forceinline__ unsigned int getBitfield(unsigned int val,
                                                    int pos,
                                                    int len) {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__ uint64_t getBitfield(uint64_t val,
                                                int pos,
                                                int len) {
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__ unsigned int setBitfield(unsigned int val,
                                                    unsigned int toInsert,
                                                    int pos,
                                                    int len) {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;"
        : "=r"(ret)
        : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ void namedBarrierWait(int name, int numThreads) {
    asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

__device__ __forceinline__ void namedBarrierArrived(int name, int numThreads) {
    asm volatile("bar.arrive %0, %1;"
                 :
                 : "r"(name), "r"(numThreads)
                 : "memory");
}

#endif  // USE_HIP

}  // namespace core
}  // namespace open3d
