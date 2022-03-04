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
// original path: faiss/faiss/gpu/utils/Limits.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <limits>

#include "open3d/core/nns/kernel/Pair.cuh"

namespace open3d {
namespace core {

template <typename T>
struct Limits {};

// Unfortunately we can't use constexpr because there is no
// constexpr constructor for half
// FIXME: faiss CPU uses +/-FLT_MAX instead of +/-infinity
constexpr float kFloatMax = std::numeric_limits<float>::max();
constexpr float kFloatMin = std::numeric_limits<float>::lowest();

template <>
struct Limits<float> {
    static __device__ __host__ inline float getMin() { return kFloatMin; }
    static __device__ __host__ inline float getMax() { return kFloatMax; }
};

constexpr double kDoubleMax = std::numeric_limits<double>::max();
constexpr double kDoubleMin = std::numeric_limits<double>::lowest();

template <>
struct Limits<double> {
    static __device__ __host__ inline double getMin() { return kDoubleMin; }
    static __device__ __host__ inline double getMax() { return kDoubleMax; }
};

constexpr int kIntMax = std::numeric_limits<int>::max();
constexpr int kIntMin = std::numeric_limits<int>::lowest();

template <>
struct Limits<int> {
    static __device__ __host__ inline int getMin() { return kIntMin; }
    static __device__ __host__ inline int getMax() { return kIntMax; }
};

template <typename K, typename V>
struct Limits<Pair<K, V>> {
    static __device__ __host__ inline Pair<K, V> getMin() {
        return Pair<K, V>(Limits<K>::getMin(), Limits<V>::getMin());
    }

    static __device__ __host__ inline Pair<K, V> getMax() {
        return Pair<K, V>(Limits<K>::getMax(), Limits<V>::getMax());
    }
};

}  // namespace core
}  // namespace open3d
