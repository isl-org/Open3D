#pragma once

#include <cassert>
#include <cmath>

#include <iostream>

typedef unsigned int uint;

namespace open3d {
static const float FLT_THRESHOLD = -1e3;
static const int FLT_PRECISION = 3;
static const int FLT_WIDTH = 8;

static const float DBL_THRESHOLD = -1e6;
static const int DBL_PRECISION = 6;
static const int DBL_WIDTH = 12;

template <typename T>
struct Vector3 {
    typedef union alignas(4 * sizeof(T)) _Type {
        // data
        struct {
            T x, y, z;
        };
        struct {
            T r, g, b;
        };

        T &operator[](const uint &i) {
            assert(i < 3);

            return ((T *)this)[i];
        }
        const T &operator[](const uint &i) const {
            assert(i < 3);

            return ((const T *const)this)[i];
        }
        operator T *const() { return reinterpret_cast<T *>(this); }
        operator const T *const() {
            return reinterpret_cast<const T *const>(this);
        }
        bool operator==(const _Type &v) const {
            return (x == v.x) && (y == v.y) && (z == v.z);
        }
        bool operator!=(const _Type &v) const { return !(this->operator==(v)); }
        bool operator<=(const _Type &v) const {
            return (x <= v.x) && (y <= v.y) && (z <= v.z);
        }
        bool operator>=(const _Type &v) const {
            return (x >= v.x) && (y >= v.y) && (z >= v.z);
        }
        _Type operator+(const _Type &v) const {
            Vector3<T>::Type output = {x + v.x, y + v.y, z + v.z};

            return output;
        }
        _Type operator-(const _Type &v) const {
            Vector3<T>::Type output = {x - v.x, y - v.y, z - v.z};

            return output;
        }
        _Type &operator+=(const _Type &v) {
            x += v.x;
            y += v.y;
            z += v.z;

            return *this;
        }
        _Type &operator-=(const _Type &v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;

            return *this;
        }
        _Type operator+(const T &t) const {
            Vector3<T>::Type output = {x + t, y + t, z + t};

            return output;
        }
        _Type operator-(const T &t) const {
            Vector3<T>::Type output = {x - t, y - t, z - t};

            return output;
        }
        _Type operator*(const T &t) const {
            Vector3<T>::Type output = {x * t, y * t, z * t};

            return output;
        }
        _Type operator/(const T &t) const {
            Vector3<T>::Type output = {x / t, y / t, z / t};

            return output;
        }
        _Type &operator+=(const T &t) {
            x += t;
            y += t;
            z += t;

            return *this;
        }
        _Type &operator-=(const T &t) {
            x -= t;
            y -= t;
            z -= t;

            return *this;
        }
        _Type &operator*=(const T &t) {
            x *= t;
            y *= t;
            z *= t;

            return *this;
        }
        _Type &operator/=(const T &t) {
            x /= t;
            y /= t;
            z /= t;

            return *this;
        }

        // Less than or equal X component.
        static bool LEX(const _Type &v0, const _Type &v1) {
            return v0.x <= v1.x;
        }
        // Greater than or equal X component.
        static bool GEX(const _Type &v0, const _Type &v1) {
            return v0.x >= v1.x;
        }
        // Less than or equal Y component.
        static bool LEY(const _Type &v0, const _Type &v1) {
            return v0.y <= v1.y;
        }
        // Greater than or equal Y component.
        static bool GEY(const _Type &v0, const _Type &v1) {
            return v0.y >= v1.y;
        }
        // Less than or equal Z component.
        static bool LEZ(const _Type &v0, const _Type &v1) {
            return v0.z <= v1.z;
        }
        // Greater than or equal Z component.
        static bool GEZ(const _Type &v0, const _Type &v1) {
            return v0.z >= v1.z;
        }
    } Type;
};

typedef Vector3<double>::Type Vector3d;
typedef Vector3<float>::Type Vector3f;
typedef Vector3<int>::Type Vector3i;

// Display.
std::ostream &operator<<(std::ostream &os, const Vector3d &v);
std::ostream &operator<<(std::ostream &os, const Vector3f &v);
std::ostream &operator<<(std::ostream &os, const Vector3i &v);
}  // namespace open3d
