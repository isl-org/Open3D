// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <cmath>

#ifdef __CUDACC__
#define FN_SPECIFIERS inline __host__ __device__
#else
#define FN_SPECIFIERS inline
#endif

namespace open3d {
namespace utility {

/// Small vector class with some basic arithmetic operations that can be used
/// within cuda kernels
template <class T, int N>
struct MiniVec {
    typedef T Scalar_t;

    FN_SPECIFIERS MiniVec() {}

    template <class... TInit>
    FN_SPECIFIERS explicit MiniVec(TInit... as) : arr{as...} {}

    FN_SPECIFIERS explicit MiniVec(const T* const ptr) {
        for (int i = 0; i < N; ++i) operator[](i) = ptr[i];
    }

    FN_SPECIFIERS const T operator[](size_t i) const { return arr[i]; }

    FN_SPECIFIERS T& operator[](size_t i) { return arr[i]; }

    template <class T2>
    FN_SPECIFIERS MiniVec<T2, N> cast() const {
        MiniVec<T2, N> a;
        for (int i = 0; i < N; ++i) a[i] = T2(operator[](i));
        return a;
    }

    FN_SPECIFIERS T dot(const MiniVec<T, N>& a) const {
        T result = 0;
        for (int i = 0; i < N; ++i) result += operator[](i) * a[i];
        return result;
    }

    FN_SPECIFIERS MiniVec<T, N> abs() const {
        MiniVec<float, N> r;
        for (int i = 0; i < N; ++i) r[i] = std::abs(operator[](i));
        return r;
    }

    FN_SPECIFIERS bool all() const {
        bool result = true;
        for (int i = 0; i < N && result; ++i) result = result && operator[](i);
        return result;
    }

    FN_SPECIFIERS bool any() const {
        for (int i = 0; i < N; ++i)
            if (operator[](i)) return true;
        return false;
    }

    T arr[N];
};

template <int N>
FN_SPECIFIERS MiniVec<float, N> floor(const MiniVec<float, N>& a) {
    MiniVec<float, N> r;
    for (int i = 0; i < N; ++i) r[i] = floorf(a[i]);
    return r;
}

template <int N>
FN_SPECIFIERS MiniVec<double, N> floor(const MiniVec<double, N>& a) {
    MiniVec<double, N> r;
    for (int i = 0; i < N; ++i) r[i] = std::floor(a[i]);
    return r;
}

template <int N>
FN_SPECIFIERS MiniVec<float, N> ceil(const MiniVec<float, N>& a) {
    MiniVec<float, N> r;
    for (int i = 0; i < N; ++i) r[i] = ceilf(a[i]);
    return r;
}

template <int N>
FN_SPECIFIERS MiniVec<double, N> ceil(const MiniVec<double, N>& a) {
    MiniVec<double, N> r;
    for (int i = 0; i < N; ++i) r[i] = std::ceil(a[i]);
    return r;
}

template <class T, int N>
FN_SPECIFIERS MiniVec<T, N> operator-(const MiniVec<T, N>& a) {
    MiniVec<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = -a[i];
    return r;
}

template <class T, int N>
FN_SPECIFIERS MiniVec<T, N> operator!(const MiniVec<T, N>& a) {
    MiniVec<T, N> r;
    for (int i = 0; i < N; ++i) r[i] = !a[i];
    return r;
}

#define DEFINE_OPERATOR(op, opas)                                          \
    template <class T, int N>                                              \
    FN_SPECIFIERS MiniVec<T, N> operator op(const MiniVec<T, N>& a,        \
                                            const MiniVec<T, N>& b) {      \
        MiniVec<T, N> c;                                                   \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b[i];                   \
        return c;                                                          \
    }                                                                      \
                                                                           \
    template <class T, int N>                                              \
    FN_SPECIFIERS void operator opas(MiniVec<T, N>& a,                     \
                                     const MiniVec<T, N>& b) {             \
        for (int i = 0; i < N; ++i) a[i] opas b[i];                        \
    }                                                                      \
                                                                           \
    template <class T, int N>                                              \
    FN_SPECIFIERS MiniVec<T, N> operator op(const MiniVec<T, N>& a, T b) { \
        MiniVec<T, N> c;                                                   \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b;                      \
        return c;                                                          \
    }                                                                      \
                                                                           \
    template <class T, int N>                                              \
    FN_SPECIFIERS MiniVec<T, N> operator op(T a, const MiniVec<T, N>& b) { \
        MiniVec<T, N> c;                                                   \
        for (int i = 0; i < N; ++i) c[i] = a op b[i];                      \
        return c;                                                          \
    }                                                                      \
                                                                           \
    template <class T, int N>                                              \
    FN_SPECIFIERS void operator opas(MiniVec<T, N>& a, T b) {              \
        for (int i = 0; i < N; ++i) a[i] opas b;                           \
    }

DEFINE_OPERATOR(+, +=)
DEFINE_OPERATOR(-, -=)
DEFINE_OPERATOR(*, *=)
DEFINE_OPERATOR(/, /=)
#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                   \
    template <class T, int N>                                                 \
    FN_SPECIFIERS MiniVec<bool, N> operator op(const MiniVec<T, N>& a,        \
                                               const MiniVec<T, N>& b) {      \
        MiniVec<bool, N> c;                                                   \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b[i];                      \
        return c;                                                             \
    }                                                                         \
                                                                              \
    template <class T, int N>                                                 \
    FN_SPECIFIERS MiniVec<bool, N> operator op(const MiniVec<T, N>& a, T b) { \
        MiniVec<T, N> c;                                                      \
        for (int i = 0; i < N; ++i) c[i] = a[i] op b;                         \
        return c;                                                             \
    }                                                                         \
                                                                              \
    template <class T, int N>                                                 \
    FN_SPECIFIERS MiniVec<T, N> operator op(T a, const MiniVec<T, N>& b) {    \
        MiniVec<bool, N> c;                                                   \
        for (int i = 0; i < N; ++i) c[i] = a op b[i];                         \
        return c;                                                             \
    }

DEFINE_OPERATOR(<)
DEFINE_OPERATOR(<=)
DEFINE_OPERATOR(>)
DEFINE_OPERATOR(>=)
DEFINE_OPERATOR(==)
DEFINE_OPERATOR(!=)
DEFINE_OPERATOR(&&)
DEFINE_OPERATOR(||)
#undef DEFINE_OPERATOR
#undef FN_SPECIFIERS

}  // namespace utility
}  // namespace open3d
