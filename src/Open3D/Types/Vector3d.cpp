
#include "Vector3d.h"

#include <cassert>

double &open3d::Vector3d::operator[](const uint &i) {
    // catch error in debug mode
    assert(i < Vector3d::COLS);

    return s[i];
}

const double &open3d::Vector3d::operator[](const uint &i) const {
    // catch error in debug mode
    assert(i < Vector3d::COLS);

    return s[i];
}

open3d::Vector3d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Vector3d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}

bool open3d::operator==(const open3d::Vector3d &v0,
                        const open3d::Vector3d &v1) {
    for (uint c = 0; c < Vector3d::COLS; c++)
        if (v0[c] != v1[c]) return false;

    return true;
}

bool open3d::operator!=(const open3d::Vector3d &v0,
                        const open3d::Vector3d &v1) {
    return !(v0 == v1);
}

bool open3d::operator<=(const open3d::Vector3d &v0,
                        const open3d::Vector3d &v1) {
    for (uint c = 0; c < Vector3d::COLS; c++)
        if (v0[c] > v1[c]) return false;

    return true;
}

bool open3d::operator>=(const open3d::Vector3d &v0,
                        const open3d::Vector3d &v1) {
    for (uint c = 0; c < Vector3d::COLS; c++)
        if (v0[c] < v1[c]) return false;

    return true;
}

open3d::Vector3d open3d::operator+(const open3d::Vector3d &v0,
                                   const open3d::Vector3d &v1) {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = v0[c] + v1[c];

    return output;
}

open3d::Vector3d open3d::operator-(const open3d::Vector3d &v0,
                                   const open3d::Vector3d &v1) {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = v0[c] - v1[c];

    return output;
}

open3d::Vector3d &open3d::operator+=(open3d::Vector3d &v0,
                                     const open3d::Vector3d &v1) {
    for (uint c = 0; c < Vector3d::COLS; c++) v0[c] += v1[c];

    return v0;
}

open3d::Vector3d &open3d::operator-=(open3d::Vector3d &v0,
                                     const open3d::Vector3d &v1) {
    for (uint c = 0; c < Vector3d::COLS; c++) v0[c] -= v1[c];

    return v0;
}

open3d::Vector3d open3d::operator+(const open3d::Vector3d &v, const double &t) {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = v[c] + t;

    return output;
}

open3d::Vector3d open3d::operator-(const open3d::Vector3d &v, const double &t) {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = v[c] - t;

    return output;
}

open3d::Vector3d open3d::operator*(const open3d::Vector3d &v, const double &t) {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = v[c] * t;

    return output;
}

open3d::Vector3d open3d::operator/(const open3d::Vector3d &v, const double &t) {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = v[c] / t;

    return output;
}

open3d::Vector3d &open3d::operator+=(open3d::Vector3d &v, const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) v[c] += t;

    return v;
}

open3d::Vector3d &open3d::operator-=(open3d::Vector3d &v, const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) v[c] -= t;

    return v;
}

open3d::Vector3d &open3d::operator*=(open3d::Vector3d &v, const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) v[c] *= t;

    return v;
}

open3d::Vector3d &open3d::operator/=(open3d::Vector3d &v, const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) v[c] /= t;

    return v;
}
