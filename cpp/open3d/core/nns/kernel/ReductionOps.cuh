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
// original path: faiss/faiss/gpu/utils/ReductionOperators.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

#include "open3d/core/nns/kernel/Limits.cuh"
#include "open3d/core/nns/kernel/Pair.cuh"

namespace open3d {
namespace core {

template <typename T>
struct Sum {
    __device__ inline T operator()(T a, T b) const { return a + b; }

    inline __device__ T identity() const { return 0.0; }
};

template <typename T>
struct Min {
    __device__ inline T operator()(T a, T b) const { return a < b ? a : b; }

    inline __device__ T identity() const { return Limits<T>::getMax(); }
};

template <typename T>
struct Max {
    __device__ inline T operator()(T a, T b) const { return a > b ? a : b; }

    inline __device__ T identity() const { return Limits<T>::getMin(); }
};

/// Used for producing segmented prefix scans; the value of the Pair
/// denotes the start of a new segment for the scan
template <typename T, typename ReduceOp>
struct SegmentedReduce {
    inline __device__ SegmentedReduce(const ReduceOp& o) : op(o) {}

    __device__ inline Pair<T, bool> operator()(const Pair<T, bool>& a,
                                               const Pair<T, bool>& b) const {
        return Pair<T, bool>(b.v ? b.k : op(a.k, b.k), a.v || b.v);
    }

    inline __device__ Pair<T, bool> identity() const {
        return Pair<T, bool>(op.identity(), false);
    }

    ReduceOp op;
};

}  // namespace core
}  // namespace open3d
