
#include "Vector3f.h"

#include <cassert>

float &open3d::Vector3f::operator[](const uint &i) {
    // catch error in debug mode
    assert(i < Vector3f::COLS);

    return s[i];
}

const float &open3d::Vector3f::operator[](const uint &i) const {
    // catch error in debug mode
    assert(i < Vector3f::COLS);

    return s[i];
}

open3d::Vector3f::operator float *const() {
    return reinterpret_cast<float *>(s);
}

open3d::Vector3f::operator const float *const() {
    return reinterpret_cast<const float *const>(s);
}

bool open3d::operator==(const open3d::Vector3f &v0,
                        const open3d::Vector3f &v1) {
    for (uint c = 0; c < Vector3f::COLS; c++)
        if (v0[c] != v1[c]) return false;

    return true;
}

bool open3d::operator!=(const open3d::Vector3f &v0,
                        const open3d::Vector3f &v1) {
    return !(v0 == v1);
}

bool open3d::operator<=(const open3d::Vector3f &v0,
                        const open3d::Vector3f &v1) {
    for (uint c = 0; c < Vector3f::COLS; c++)
        if (v0[c] > v1[c]) return false;

    return true;
}

bool open3d::operator>=(const open3d::Vector3f &v0,
                        const open3d::Vector3f &v1) {
    for (uint c = 0; c < Vector3f::COLS; c++)
        if (v0[c] < v1[c]) return false;

    return true;
}

open3d::Vector3f open3d::operator+(const open3d::Vector3f &v0,
                                   const open3d::Vector3f &v1) {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = v0[c] + v1[c];

    return output;
}

open3d::Vector3f open3d::operator-(const open3d::Vector3f &v0,
                                   const open3d::Vector3f &v1) {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = v0[c] - v1[c];

    return output;
}

open3d::Vector3f &open3d::operator+=(open3d::Vector3f &v0,
                                     const open3d::Vector3f &v1) {
    for (uint c = 0; c < Vector3f::COLS; c++) v0[c] += v1[c];

    return v0;
}

open3d::Vector3f &open3d::operator-=(open3d::Vector3f &v0,
                                     const open3d::Vector3f &v1) {
    for (uint c = 0; c < Vector3f::COLS; c++) v0[c] -= v1[c];

    return v0;
}

open3d::Vector3f open3d::operator+(const open3d::Vector3f &v, const float &t) {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = v[c] + t;

    return output;
}

open3d::Vector3f open3d::operator-(const open3d::Vector3f &v, const float &t) {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = v[c] - t;

    return output;
}

open3d::Vector3f open3d::operator*(const open3d::Vector3f &v, const float &t) {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = v[c] * t;

    return output;
}

open3d::Vector3f open3d::operator/(const open3d::Vector3f &v, const float &t) {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = v[c] / t;

    return output;
}

open3d::Vector3f &open3d::operator+=(open3d::Vector3f &v, const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) v[c] += t;

    return v;
}

open3d::Vector3f &open3d::operator-=(open3d::Vector3f &v, const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) v[c] -= t;

    return v;
}

open3d::Vector3f &open3d::operator*=(open3d::Vector3f &v, const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) v[c] *= t;

    return v;
}

open3d::Vector3f &open3d::operator/=(open3d::Vector3f &v, const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) v[c] /= t;

    return v;
}
