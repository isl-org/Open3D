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

#include "Matrix3d.h"
#include "Matrix3f.h"

#include "Matrix4d.h"
#include "Matrix4f.h"

#include "Matrix6d.h"
#include "Matrix6f.h"

#include "Vector3d.h"
#include "Vector3f.h"
#include "Vector3i.h"

namespace open3d {
template <typename T>
bool operator==(const T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++)
            if (m0.s[r][c] != m1.s[r][c]) return false;

    return true;
}
template <typename T>
bool operator!=(const T &m0, const T &m1) {
    return !(m0 == m1);
}
template <typename T>
bool operator<=(const T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++)
            if (m0.s[r][c] > m1.s[r][c]) return false;

    return true;
}
template <typename T>
bool operator>=(const T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++)
            if (m0.s[r][c] < m1.s[r][c]) return false;

    return true;
}
template <typename T>
T operator+(const T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    T output;
    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++)
            output.s[r][c] = m0.s[r][c] + m1.s[r][c];

    return output;
}
template <typename T>
T operator-(const T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    T output;
    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++)
            output.s[r][c] = m0.s[r][c] - m1.s[r][c];

    return output;
}
template <typename T>
T &operator+=(T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) m0.s[r][c] += m1.s[r][c];

    return m0;
}
template <typename T>
T &operator-=(T &m0, const T &m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) m0.s[r][c] -= m1.s[r][c];

    return m0;
}
template <typename T>
T operator+(const T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    T output;
    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) output.s[r][c] = m.s[r][c] + t;

    return output;
}
template <typename T>
T operator-(const T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    T output;
    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) output.s[r][c] = m.s[r][c] - t;

    return output;
}
template <typename T>
T operator*(const T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    T output;
    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) output.s[r][c] = m.s[r][c] * t;

    return output;
}
template <typename T>
T operator/(const T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    T output;
    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) output.s[r][c] = m.s[r][c] / t;

    return output;
}
template <typename T>
T &operator+=(T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) m.s[r][c] += t;

    return m;
}
template <typename T>
T &operator-=(T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) m.s[r][c] -= t;

    return m;
}
template <typename T>
T &operator*=(T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) m.s[r][c] *= t;

    return m;
}
template <typename T>
T &operator/=(T &m, const float &t) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value ||
                          std::is_base_of<Vector3d, T>::value ||
                          std::is_base_of<Vector3f, T>::value ||
                          std::is_base_of<Vector3i, T>::value ||
                          std::is_base_of<Vector4d, T>::value ||
                          std::is_base_of<Vector4f, T>::value ||
                          std::is_base_of<Vector6d, T>::value ||
                          std::is_base_of<Vector6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < T::ROWS; r++)
        for (uint c = 0; c < T::COLS; c++) m.s[r][c] /= t;

    return m;
}
}  // namespace open3d
