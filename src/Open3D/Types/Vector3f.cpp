
#include "Vector3f.h"

#include <cassert>

float &open3d::Vector3f::operator[](const uint &i) {
    // catch error in debug mode
    assert(0 <= i && i < Vector3f::COLS);

    return s[i];
}

const float &open3d::Vector3f::operator[](const uint &i) const {
    // catch error in debug mode
    assert(0 <= i && i < Vector3f::COLS);

    return s[i];
}

open3d::Vector3f::operator float *const() {
    return reinterpret_cast<float *>(s);
}

open3d::Vector3f::operator const float *const() {
    return reinterpret_cast<const float *const>(s);
}

bool open3d::Vector3f::operator==(const typename open3d::Vector3f &v) {
    for (uint c = 0; c < Vector3f::COLS; c++)
        if ((*this)[c] != v[c]) return false;

    return true;
}

bool open3d::Vector3f::operator!=(const typename open3d::Vector3f &v) {
    return !(*this == v);
}

bool open3d::Vector3f::operator<=(const typename open3d::Vector3f &v) {
    for (uint c = 0; c < Vector3f::COLS; c++)
        if ((*this)[c] > v[c]) return false;

    return true;
}

bool open3d::Vector3f::operator>=(const typename open3d::Vector3f &v) {
    for (uint c = 0; c < Vector3f::COLS; c++)
        if ((*this)[c] < v[c]) return false;

    return true;
}

typename open3d::Vector3f open3d::Vector3f::operator+(
        const typename open3d::Vector3f &v) const {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = (*this)[c] + v[c];

    return output;
}

typename open3d::Vector3f open3d::Vector3f::operator-(
        const typename open3d::Vector3f &v) const {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = (*this)[c] - v[c];

    return output;
}

typename open3d::Vector3f &open3d::Vector3f::operator+=(
        const typename open3d::Vector3f &v) {
    for (uint c = 0; c < Vector3f::COLS; c++) (*this)[c] += v[c];

    return *this;
}

typename open3d::Vector3f &open3d::Vector3f::operator-=(
        const typename open3d::Vector3f &v) {
    for (uint c = 0; c < Vector3f::COLS; c++) (*this)[c] -= v[c];

    return *this;
}

typename open3d::Vector3f open3d::Vector3f::operator+(const float &t) const {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = (*this)[c] + t;

    return output;
}

typename open3d::Vector3f open3d::Vector3f::operator-(const float &t) const {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = (*this)[c] - t;

    return output;
}

typename open3d::Vector3f open3d::Vector3f::operator*(const float &t) const {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = (*this)[c] * t;

    return output;
}

typename open3d::Vector3f open3d::Vector3f::operator/(const float &t) const {
    Vector3f output;
    for (uint c = 0; c < Vector3f::COLS; c++) output[c] = (*this)[c] / t;

    return output;
}

typename open3d::Vector3f &open3d::Vector3f::operator+=(const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) (*this)[c] += t;

    return *this;
}

typename open3d::Vector3f &open3d::Vector3f::operator-=(const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) (*this)[c] -= t;

    return *this;
}

typename open3d::Vector3f &open3d::Vector3f::operator*=(const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) (*this)[c] *= t;

    return *this;
}

typename open3d::Vector3f &open3d::Vector3f::operator/=(const float &t) {
    for (uint c = 0; c < Vector3f::COLS; c++) (*this)[c] /= t;

    return *this;
}
