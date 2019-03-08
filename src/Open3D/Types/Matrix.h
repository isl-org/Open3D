#pragma once

#include <cassert>
#include <cmath>

#include <iostream>

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
template <typename T, uint ROWS, uint COLS>
struct Matrix {
    typedef struct _Type {
        T s[ROWS][COLS];

        // subscript operator: readwrite
        T *operator[](const uint &i) {
            assert(i < ROWS);

            return (T *)&s[i];
        }
        // subscript operator: readonly
        const T *operator[](const uint &i) const {
            assert(i < ROWS);

            return (const T *const) & s[i];
        }

        // casting operator: readwrite
        explicit operator T *const() { return reinterpret_cast<T *>(s); }
        // casting operator: readonly
        explicit operator const T *const() {
            return reinterpret_cast<const T *const>(s);
        }

        bool operator==(const _Type &m) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    if ((*this)[r][c] != m[r][c]) return false;

            return true;
        }
        bool operator!=(const _Type &m) { return !(*this == m); }
        bool operator<=(const _Type &m) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    if ((*this)[r][c] > m[r][c]) return false;

            return true;
        }
        bool operator>=(const _Type &m) {
            for (uint r = 0; r < ROWS; r++)
                if ((*this)[r] < m[r]) return false;

            return true;
        }

        // addition
        _Type operator+(const _Type &m) const {
            Matrix<T, ROWS, COLS>::Type output;
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = (*this)[r][c] + m[r][c];

            return output;
        }
        // subtraction
        _Type operator-(const _Type &m) const {
            Matrix<T, ROWS, COLS>::Type output;
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = (*this)[r][c] - m[r][c];

            return output;
        }
        // addition assignment
        _Type &operator+=(const _Type &m) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++) (*this)[r][c] += m[r][c];

            return *this;
        }
        // subtraction assignment
        _Type &operator-=(const _Type &m) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++) (*this)[r][c] -= m[r][c];

            return *this;
        }
        // addition
        _Type operator+(const T &t) const {
            Matrix<T, ROWS, COLS>::Type output;
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = (*this)[r][c] + t;

            return output;
        }
        // subtraction
        _Type operator-(const T &t) const {
            Matrix<T, ROWS, COLS>::Type output;
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = (*this)[r][c] - t;

            return output;
        }
        // multiply with scalar
        _Type operator*(const T &t) const {
            Matrix<T, ROWS, COLS>::Type output;
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = (*this)[r][c] * t;

            return output;
        }
        // divide by scalar
        _Type operator/(const T &t) const {
            Matrix<T, ROWS, COLS>::Type output;
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++)
                    output[r][c] = (*this)[r][c] / t;

            return output;
        }
        // addition assignment
        _Type &operator+=(const T &t) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++) (*this)[r][c] += t;

            return *this;
        }
        // subtraction assignment
        _Type &operator-=(const T &t) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++) (*this)[r][c] -= t;

            return *this;
        }
        // multiplication assignment
        _Type &operator*=(const T &t) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++) (*this)[r][c] *= t;

            return *this;
        }
        // division assignment
        _Type &operator/=(const T &t) {
            for (uint r = 0; r < ROWS; r++)
                for (uint c = 0; c < COLS; c++) (*this)[r][c] /= t;

            return *this;
        }
    } Type;
};

// 1D tensor, row major
template <typename T, uint COLS>
struct Matrix<T, 1, COLS> {
    typedef struct _Type {
        T s[COLS];

        // subscript operator: readwrite
        T &operator[](const uint &i) {
            assert(i < COLS);

            return s[i];
        }
        // subscript operator: readonly
        const T &operator[](const uint &i) const {
            assert(i < COLS);

            return s[i];
        }

        // casting operator: readwrite
        explicit operator T *const() { return reinterpret_cast<T *>(s); }
        // casting operator: readonly
        explicit operator const T *const() {
            return reinterpret_cast<const T *const>(s);
        }

        bool operator==(const _Type &v) {
            for (uint c = 0; c < COLS; c++)
                if ((*this)[c] != v[c]) return false;

            return true;
        }
        bool operator!=(const _Type &v) { return !(*this == v); }
        bool operator<=(const _Type &v) {
            for (uint c = 0; c < COLS; c++)
                if ((*this)[c] > v[c]) return false;

            return true;
        }
        bool operator>=(const _Type &v) {
            for (uint c = 0; c < COLS; c++)
                if ((*this)[c] < v[c]) return false;

            return true;
        }

        // addition
        _Type operator+(const _Type &v) const {
            Matrix<T, 1, COLS>::Type output;
            for (uint c = 0; c < COLS; c++) output[c] = (*this)[c] + v[c];

            return output;
        }
        // subtraction
        _Type operator-(const _Type &v) const {
            Matrix<T, 1, COLS>::Type output;
            for (uint c = 0; c < COLS; c++) output[c] = (*this)[c] - v[c];

            return output;
        }
        // addition assignment
        _Type &operator+=(const _Type &v) {
            for (uint c = 0; c < COLS; c++) (*this)[c] += v[c];

            return *this;
        }
        // subtraction assignment
        _Type &operator-=(const _Type &v) {
            for (uint c = 0; c < COLS; c++) (*this)[c] -= v[c];

            return *this;
        }
        // addition
        _Type operator+(const T &t) const {
            Matrix<T, 1, COLS>::Type output;
            for (uint c = 0; c < COLS; c++) output[c] = (*this)[c] + t;

            return output;
        }
        // subtraction
        _Type operator-(const T &t) const {
            Matrix<T, 1, COLS>::Type output;
            for (uint c = 0; c < COLS; c++) output[c] = (*this)[c] - t;

            return output;
        }
        // multiply with scalar
        _Type operator*(const T &t) const {
            Matrix<T, 1, COLS>::Type output;
            for (uint c = 0; c < COLS; c++) output[c] = (*this)[c] * t;

            return output;
        }
        // divide by scalar
        _Type operator/(const T &t) const {
            Matrix<T, 1, COLS>::Type output;
            for (uint c = 0; c < COLS; c++) output[c] = (*this)[c] / t;

            return output;
        }
        // addition assignment
        _Type &operator+=(const T &t) {
            for (uint c = 0; c < COLS; c++) (*this)[c] += t;

            return *this;
        }
        // subtraction assignment
        _Type &operator-=(const T &t) {
            for (uint c = 0; c < COLS; c++) (*this)[c] -= t;

            return *this;
        }
        // multiplication assignment
        _Type &operator*=(const T &t) {
            for (uint c = 0; c < COLS; c++) (*this)[c] *= t;

            return *this;
        }
        // division assignment
        _Type &operator/=(const T &t) {
            for (uint c = 0; c < COLS; c++) (*this)[c] /= t;

            return *this;
        }
    } Type;
};

// 2D tensor, row major, size 3x3
template <typename T>
using Matrix3 = typename Matrix<T, 3, 3>::Type;
// 2D tensor, row major, size 4x4
template <typename T>
using Matrix4 = typename Matrix<T, 4, 4>::Type;
// 2D tensor, row major, size 6x6
template <typename T>
using Matrix6 = typename Matrix<T, 6, 6>::Type;

// 1D tensor, row major, size 3
template <typename T>
using Vector3 = typename Matrix<T, 1, 3>::Type;

// 2D 3x3 tensor
typedef Matrix3<double> Matrix3d;
typedef Matrix3<float> Matrix3f;

// 2D 4x4 tensor
typedef Matrix4<double> Matrix4d;
typedef Matrix4<float> Matrix4f;

// 2D 6x6 tensor
typedef Matrix6<double> Matrix6d;
typedef Matrix6<float> Matrix6f;

// 1D 1x3 tensor
typedef Vector3<double> Vector3d;
typedef Vector3<float> Vector3f;
typedef Vector3<int> Vector3i;
}  // namespace open3d
