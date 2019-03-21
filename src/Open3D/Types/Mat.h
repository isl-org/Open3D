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

#include "setup.h"

#include <cassert>

namespace open3d {
// 1D tensor, row major
template <typename T, uint COLS>
struct Vec {
    typedef struct _Type {
        static const uint Rows = 1;
        static const uint Cols = COLS;
        static const uint Size = Rows * Cols;

        T s[COLS];

        // subscript operator: readwrite
        OPEN3D_FUNC_DECL inline T &operator[](const uint &i) {
            // catch error in debug mode
            assert(i < Cols);

            return s[i];
        }
        // subscript operator: readonly
        OPEN3D_FUNC_DECL inline const T &operator[](const uint &i) const {
            // catch error in debug mode
            assert(i < Cols);

            return s[i];
        }
        OPEN3D_FUNC_DECL inline bool operator==(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++)
                if (s[c] != v[c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator!=(const _Type &v) {
            return !(*this == v);
        }
        OPEN3D_FUNC_DECL inline bool operator<=(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++)
                if (s[c] > v[c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator>=(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++)
                if (s[c] < v[c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator<(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++)
                if (s[c] >= v[c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator>(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++)
                if (s[c] <= v[c]) return false;

            return true;
        }
        // addition
        OPEN3D_FUNC_DECL inline _Type operator+(const _Type &v) const {
            _Type output;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output[c] = s[c] + v[c];

            return output;
        }
        // subtraction
        OPEN3D_FUNC_DECL inline _Type operator-(const _Type &v) const {
            _Type output;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output[c] = s[c] - v[c];

            return output;
        }
        // addition assignment
        OPEN3D_FUNC_DECL inline _Type &operator+=(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] += v[c];

            return *this;
        }
        // subtraction assignment
        OPEN3D_FUNC_DECL inline _Type &operator-=(const _Type &v) {
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] -= v[c];

            return *this;
        }
        // addition
        OPEN3D_FUNC_DECL inline _Type operator+(const T &t) const {
            _Type output;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output[c] = s[c] + t;

            return output;
        }
        // subtraction
        OPEN3D_FUNC_DECL inline _Type operator-(const T &t) const {
            _Type output;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output[c] = s[c] - t;

            return output;
        }
        // multiply with scalar
        OPEN3D_FUNC_DECL inline _Type operator*(const T &t) const {
            _Type output;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output[c] = s[c] * t;

            return output;
        }
        // divide by scalar
        OPEN3D_FUNC_DECL inline _Type operator/(const T &t) const {
            _Type output;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output[c] = s[c] / t;

            return output;
        }
        // addition assignment
        OPEN3D_FUNC_DECL inline _Type &operator+=(const T &t) {
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] += t;

            return *this;
        }
        // subtraction assignment
        OPEN3D_FUNC_DECL inline _Type &operator-=(const T &t) {
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] -= t;

            return *this;
        }
        // multiplication assignment
        OPEN3D_FUNC_DECL inline _Type &operator*=(const T &t) {
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] *= t;

            return *this;
        }
        // division assignment
        OPEN3D_FUNC_DECL inline _Type &operator/=(const T &t) {
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] /= t;

            return *this;
        }
        OPEN3D_FUNC_DECL inline T squaredNorm() const {
            T output = (T)0;
#pragma unroll
            for (uint c = 0; c < COLS; c++) output += s[c] * s[c];

            return output;
        }
        OPEN3D_FUNC_DECL inline void normalize() {
            T norm = sqrt(squaredNorm());
#pragma unroll
            for (uint c = 0; c < COLS; c++) s[c] /= norm;
        }
        OPEN3D_FUNC_DECL inline _Type cross(const _Type &v) const {
            static_assert(1 == Rows, "must be a 1x3 vector");
            static_assert(3 == COLS, "must be a 1x3 vector");

            _Type output{};
            output[0] = s[1] * v.s[2] - s[2] * v.s[1];
            output[1] = s[2] * v.s[0] - s[0] * v.s[2];
            output[2] = s[0] * v.s[1] - s[1] * v.s[0];

            return output;
        }

    } Type;
};

// 2D tensor, row major
template <typename T, uint ROWS, uint COLS>
struct Mat {
    typedef struct _Type {
        static const uint Rows = ROWS;
        static const uint Cols = COLS;
        static const uint Size = Rows * Cols;

        typename Vec<T, COLS>::Type s[ROWS];

        // subscript operator: readwrite
        OPEN3D_FUNC_DECL inline typename Vec<T, COLS>::Type &operator[](
                const uint &i) {
            // catch error in debug mode
            assert(i < Rows);

            return s[i];
        }
        // subscript operator: readonly
        OPEN3D_FUNC_DECL inline const typename Vec<T, COLS>::Type &operator[](
                const uint &i) const {
            // catch error in debug mode
            assert(i < Rows);

            return s[i];
        }
        OPEN3D_FUNC_DECL inline bool operator==(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    if (s[r][c] != m.s[r][c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator!=(const _Type &m) {
            return !(*this == m);
        }
        OPEN3D_FUNC_DECL inline bool operator<=(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    if (s[r][c] > m.s[r][c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator>=(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    if (s[r][c] < m.s[r][c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator<(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    if (s[r][c] >= m.s[r][c]) return false;

            return true;
        }
        OPEN3D_FUNC_DECL inline bool operator>(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    if (s[r][c] <= m.s[r][c]) return false;

            return true;
        }
        // addition
        OPEN3D_FUNC_DECL inline _Type operator+(const _Type &m) const {
            _Type output;
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = s[r][c] + m.s[r][c];

            return output;
        }
        // subtraction
        OPEN3D_FUNC_DECL inline _Type operator-(const _Type &m) const {
            _Type output;
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = s[r][c] - m.s[r][c];

            return output;
        }
        // addition assignment
        OPEN3D_FUNC_DECL inline _Type &operator+=(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) s[r][c] += m.s[r][c];

            return *this;
        }
        // subtraction assignment
        OPEN3D_FUNC_DECL inline _Type &operator-=(const _Type &m) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) s[r][c] -= m.s[r][c];

            return *this;
        }
        // addition
        OPEN3D_FUNC_DECL inline _Type operator+(const T &t) const {
            _Type output;
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) output[r][c] = s[r][c] + t;

            return output;
        }
        // subtraction
        OPEN3D_FUNC_DECL inline _Type operator-(const T &t) const {
            _Type output;
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) output[r][c] = s[r][c] - t;

            return output;
        }
        // multiply with scalar
        OPEN3D_FUNC_DECL inline _Type operator*(const T &t) const {
            _Type output;
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) output[r][c] = s[r][c] * t;

            return output;
        }
        // divide by scalar
        OPEN3D_FUNC_DECL inline _Type operator/(const T &t) const {
            _Type output;
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) output[r][c] = s[r][c] / t;

            return output;
        }
        // addition assignment
        OPEN3D_FUNC_DECL inline _Type &operator+=(const T &t) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) s[r][c] += t;

            return *this;
        }
        // subtraction assignment
        OPEN3D_FUNC_DECL inline _Type &operator-=(const T &t) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) s[r][c] -= t;

            return *this;
        }
        // multiplication assignment
        OPEN3D_FUNC_DECL inline _Type &operator*=(const T &t) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) s[r][c] *= t;

            return *this;
        }
        // division assignment
        OPEN3D_FUNC_DECL inline _Type &operator/=(const T &t) {
#pragma unroll
            for (uint r = 0; r < ROWS; r++)
#pragma unroll
                for (uint c = 0; c < COLS; c++) s[r][c] /= t;

            return *this;
        }
    } Type;
};

// 1D tensor, row major
template <typename T, uint COLS>
using VecType = typename Vec<T, COLS>::Type;
// 1D tensor, row major, size 2
template <typename T>
using Vec2 = typename Vec<T, 2>::Type;
// 1D tensor, row major, size 3
template <typename T>
using Vec3 = typename Vec<T, 3>::Type;
// 1D tensor, row major, size 4
template <typename T>
using Vec4 = typename Vec<T, 4>::Type;
// 1D tensor, row major, size 6
template <typename T>
using Vec6 = typename Vec<T, 6>::Type;

// 1D 1x2 tensor
typedef Vec2<int> Vec2i;

// 1D 1x3 tensor
typedef Vec3<double> Vec3d;
typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;

// 1D 1x4 tensor
typedef Vec4<double> Vec4d;
typedef Vec4<float> Vec4f;

// 1D 1x6 tensor
typedef Vec6<double> Vec6d;
typedef Vec6<float> Vec6f;

// 2D tensor, row major
template <typename T, uint ROWS, uint COLS>
using MatType = typename Mat<T, ROWS, COLS>::Type;
// 2D tensor, row major, size 3x3
template <typename T>
using Mat3 = typename Mat<T, 3, 3>::Type;
// 2D tensor, row major, size 4x4
template <typename T>
using Mat4 = typename Mat<T, 4, 4>::Type;
// 2D tensor, row major, size 6x6
template <typename T>
using Mat6 = typename Mat<T, 6, 6>::Type;

// 2D 3x3 tensor
typedef Mat3<double> Mat3d;
typedef Mat3<float> Mat3f;

// 2D 4x4 tensor
typedef Mat4<double> Mat4d;
typedef Mat4<float> Mat4f;

// 2D 6x6 tensor
typedef Mat6<double> Mat6d;
typedef Mat6<float> Mat6f;
}  // namespace open3d
