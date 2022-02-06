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
// original path: faiss/faiss/gpu/utils/Pair.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

#include "open3d/core/nns/kernel/PtxUtils.cuh"
#include "open3d/core/nns/kernel/WarpShuffle.cuh"

namespace open3d {
namespace core {

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
    constexpr __device__ inline Pair() {}

    constexpr __device__ inline Pair(K key, V value) : k(key), v(value) {}

    __device__ inline bool operator==(const Pair<K, V>& rhs) const {
        return k == rhs.k && v == rhs.v;
    }

    __device__ inline bool operator!=(const Pair<K, V>& rhs) const {
        return !operator==(rhs);
    }

    __device__ inline bool operator<(const Pair<K, V>& rhs) const {
        return k < rhs.k || (k == rhs.k && v < rhs.v);
    }

    __device__ inline bool operator>(const Pair<K, V>& rhs) const {
        return k > rhs.k || (k == rhs.k && v > rhs.v);
    }

    K k;
    V v;
};

template <typename T, typename U>
inline __device__ Pair<T, U> shfl_up(const Pair<T, U>& pair,
                                     unsigned int delta,
                                     int width = kWarpSize) {
    return Pair<T, U>(shfl_up(pair.k, delta, width),
                      shfl_up(pair.v, delta, width));
}

template <typename T, typename U>
inline __device__ Pair<T, U> shfl_xor(const Pair<T, U>& pair,
                                      int laneMask,
                                      int width = kWarpSize) {
    return Pair<T, U>(shfl_xor(pair.k, laneMask, width),
                      shfl_xor(pair.v, laneMask, width));
}

}  // namespace core
}  // namespace open3d
