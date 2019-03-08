
#include "Vector3i.h"

#include <cassert>

int &open3d::Vector3i::operator[](const uint &i) {
    // catch error in debug mode
    assert(i < Vector3i::COLS);

    return s[i];
}

const int &open3d::Vector3i::operator[](const uint &i) const {
    // catch error in debug mode
    assert(i < Vector3i::COLS);

    return s[i];
}

open3d::Vector3i::operator int *const() {
    return reinterpret_cast<int *>(s);
}

open3d::Vector3i::operator const int *const() {
    return reinterpret_cast<const int *const>(s);
}

bool open3d::operator==(const open3d::Vector3i &v0,
                        const open3d::Vector3i &v1) {
    for (uint c = 0; c < Vector3i::COLS; c++)
        if (v0[c] != v1[c]) return false;

    return true;
}

bool open3d::operator!=(const open3d::Vector3i &v0,
                        const open3d::Vector3i &v1) {
    return !(v0 == v1);
}

bool open3d::operator<=(const open3d::Vector3i &v0,
                        const open3d::Vector3i &v1) {
    for (uint c = 0; c < Vector3i::COLS; c++)
        if (v0[c] > v1[c]) return false;

    return true;
}

bool open3d::operator>=(const open3d::Vector3i &v0,
                        const open3d::Vector3i &v1) {
    for (uint c = 0; c < Vector3i::COLS; c++)
        if (v0[c] < v1[c]) return false;

    return true;
}

open3d::Vector3i open3d::operator+(const open3d::Vector3i &v0,
                                   const open3d::Vector3i &v1) {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = v0[c] + v1[c];

    return output;
}

open3d::Vector3i open3d::operator-(const open3d::Vector3i &v0,
                                   const open3d::Vector3i &v1) {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = v0[c] - v1[c];

    return output;
}

open3d::Vector3i &open3d::operator+=(open3d::Vector3i &v0,
                                     const open3d::Vector3i &v1) {
    for (uint c = 0; c < Vector3i::COLS; c++) v0[c] += v1[c];

    return v0;
}

open3d::Vector3i &open3d::operator-=(open3d::Vector3i &v0,
                                     const open3d::Vector3i &v1) {
    for (uint c = 0; c < Vector3i::COLS; c++) v0[c] -= v1[c];

    return v0;
}

open3d::Vector3i open3d::operator+(const open3d::Vector3i &v, const int &t) {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = v[c] + t;

    return output;
}

open3d::Vector3i open3d::operator-(const open3d::Vector3i &v, const int &t) {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = v[c] - t;

    return output;
}

open3d::Vector3i open3d::operator*(const open3d::Vector3i &v, const int &t) {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = v[c] * t;

    return output;
}

open3d::Vector3i open3d::operator/(const open3d::Vector3i &v, const int &t) {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = v[c] / t;

    return output;
}

open3d::Vector3i &open3d::operator+=(open3d::Vector3i &v, const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) v[c] += t;

    return v;
}

open3d::Vector3i &open3d::operator-=(open3d::Vector3i &v, const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) v[c] -= t;

    return v;
}

open3d::Vector3i &open3d::operator*=(open3d::Vector3i &v, const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) v[c] *= t;

    return v;
}

open3d::Vector3i &open3d::operator/=(open3d::Vector3i &v, const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) v[c] /= t;

    return v;
}
