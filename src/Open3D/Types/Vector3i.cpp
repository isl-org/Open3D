
#include "Vector3i.h"

#include <cassert>

int &open3d::Vector3i::operator[](const uint &i) {
    // catch error in debug mode
    assert(0 <= i && i < Vector3i::COLS);

    return s[i];
}

const int &open3d::Vector3i::operator[](const uint &i) const {
    // catch error in debug mode
    assert(0 <= i && i < Vector3i::COLS);

    return s[i];
}

open3d::Vector3i::operator int *const() {
    return reinterpret_cast<int *>(s);
}

open3d::Vector3i::operator const int *const() {
    return reinterpret_cast<const int *const>(s);
}

bool open3d::Vector3i::operator==(const typename open3d::Vector3i &v) {
    for (uint c = 0; c < Vector3i::COLS; c++)
        if ((*this)[c] != v[c]) return false;

    return true;
}

bool open3d::Vector3i::operator!=(const typename open3d::Vector3i &v) {
    return !(*this == v);
}

bool open3d::Vector3i::operator<=(const typename open3d::Vector3i &v) {
    for (uint c = 0; c < Vector3i::COLS; c++)
        if ((*this)[c] > v[c]) return false;

    return true;
}

bool open3d::Vector3i::operator>=(const typename open3d::Vector3i &v) {
    for (uint c = 0; c < Vector3i::COLS; c++)
        if ((*this)[c] < v[c]) return false;

    return true;
}

typename open3d::Vector3i open3d::Vector3i::operator+(
        const typename open3d::Vector3i &v) const {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = (*this)[c] + v[c];

    return output;
}

typename open3d::Vector3i open3d::Vector3i::operator-(
        const typename open3d::Vector3i &v) const {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = (*this)[c] - v[c];

    return output;
}

typename open3d::Vector3i &open3d::Vector3i::operator+=(
        const typename open3d::Vector3i &v) {
    for (uint c = 0; c < Vector3i::COLS; c++) (*this)[c] += v[c];

    return *this;
}

typename open3d::Vector3i &open3d::Vector3i::operator-=(
        const typename open3d::Vector3i &v) {
    for (uint c = 0; c < Vector3i::COLS; c++) (*this)[c] -= v[c];

    return *this;
}

typename open3d::Vector3i open3d::Vector3i::operator+(const int &t) const {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = (*this)[c] + t;

    return output;
}

typename open3d::Vector3i open3d::Vector3i::operator-(const int &t) const {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = (*this)[c] - t;

    return output;
}

typename open3d::Vector3i open3d::Vector3i::operator*(const int &t) const {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = (*this)[c] * t;

    return output;
}

typename open3d::Vector3i open3d::Vector3i::operator/(const int &t) const {
    Vector3i output;
    for (uint c = 0; c < Vector3i::COLS; c++) output[c] = (*this)[c] / t;

    return output;
}

typename open3d::Vector3i &open3d::Vector3i::operator+=(const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) (*this)[c] += t;

    return *this;
}

typename open3d::Vector3i &open3d::Vector3i::operator-=(const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) (*this)[c] -= t;

    return *this;
}

typename open3d::Vector3i &open3d::Vector3i::operator*=(const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) (*this)[c] *= t;

    return *this;
}

typename open3d::Vector3i &open3d::Vector3i::operator/=(const int &t) {
    for (uint c = 0; c < Vector3i::COLS; c++) (*this)[c] /= t;

    return *this;
}
