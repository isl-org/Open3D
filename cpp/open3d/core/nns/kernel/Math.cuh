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
#pragma once

//
// Templated wrappers to express math for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace open3d {
namespace core {

template <typename T>
struct Math {
    typedef T ScalarType;

    static inline __device__ T add(T a, T b) { return a + b; }

    static inline __device__ T sub(T a, T b) { return a - b; }

    static inline __device__ T mul(T a, T b) { return a * b; }

    static inline __device__ T neg(T v) { return -v; }

    /// For a vector type, this is a horizontal add, returning sum(v_i)
    // static inline __device__ float reduceAdd(T v) {
    //     return ConvertTo<float>::to(v);
    // }

    static inline __device__ bool lt(T a, T b) { return a < b; }

    static inline __device__ bool gt(T a, T b) { return a > b; }

    static inline __device__ bool eq(T a, T b) { return a == b; }

    static inline __device__ T zero() { return (T)0; }
};

}  // namespace core
}  // namespace open3d