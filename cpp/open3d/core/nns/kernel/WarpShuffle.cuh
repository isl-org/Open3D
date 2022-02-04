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
// original path: faiss/faiss/gpu/utils/WarpShuffles.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

namespace open3d {
namespace core {

// defines to simplify the SASS assembly structure file/line in the profiler
#if CUDA_VERSION >= 9000
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) \
    __shfl_sync(0xffffffff, VAL, SRC_LANE, WIDTH)
#else
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) __shfl(VAL, SRC_LANE, WIDTH)
#endif

template <typename T>
inline __device__ T shfl(const T val, int srcLane, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_sync(0xffffffff, val, srcLane, width);
#else
    return __shfl(val, srcLane, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl(T* const val, int srcLane, int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T shfl_up(const T val,
                            unsigned int delta,
                            int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_up_sync(0xffffffff, val, delta, width);
#else
    return __shfl_up(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_up(T* const val,
                             unsigned int delta,
                             int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl_up(v, delta, width);
}

template <typename T>
inline __device__ T shfl_down(const T val,
                              unsigned int delta,
                              int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(0xffffffff, val, delta, width);
#else
    return __shfl_down(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_down(T* const val,
                               unsigned int delta,
                               int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
    return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(T* const val,
                              int laneMask,
                              int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_xor(v, laneMask, width);
}

// CUDA 9.0+ has half shuffle
#if CUDA_VERSION < 9000
inline __device__ half shfl(half v, int srcLane, int width = kWarpSize) {
    unsigned int vu = v.x;
    vu = __shfl(vu, srcLane, width);

    half h;
    h.x = (unsigned short)vu;
    return h;
}

inline __device__ half shfl_xor(half v, int laneMask, int width = kWarpSize) {
    unsigned int vu = v.x;
    vu = __shfl_xor(vu, laneMask, width);

    half h;
    h.x = (unsigned short)vu;
    return h;
}
#endif  // CUDA_VERSION

}  // namespace core
}  // namespace open3d
