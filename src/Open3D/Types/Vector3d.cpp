
#include "Vector3d.h"

#include <cassert>

double &open3d::Vector3d::operator[](const uint &i) {
    // catch error in debug mode
    assert(0 <= i && i < Vector3d::COLS);

    return s[i];
}

const double &open3d::Vector3d::operator[](const uint &i) const {
    // catch error in debug mode
    assert(0 <= i && i < Vector3d::COLS);

    return s[i];
}

open3d::Vector3d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Vector3d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}

bool open3d::Vector3d::operator==(const typename open3d::Vector3d &v) {
    for (uint c = 0; c < Vector3d::COLS; c++)
        if ((*this)[c] != v[c]) return false;

    return true;
}

bool open3d::Vector3d::operator!=(const typename open3d::Vector3d &v) {
    return !(*this == v);
}

bool open3d::Vector3d::operator<=(const typename open3d::Vector3d &v) {
    for (uint c = 0; c < Vector3d::COLS; c++)
        if ((*this)[c] > v[c]) return false;

    return true;
}

bool open3d::Vector3d::operator>=(const typename open3d::Vector3d &v) {
    for (uint c = 0; c < Vector3d::COLS; c++)
        if ((*this)[c] < v[c]) return false;

    return true;
}

typename open3d::Vector3d open3d::Vector3d::operator+(
        const typename open3d::Vector3d &v) const {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = (*this)[c] + v[c];

    return output;
}

typename open3d::Vector3d open3d::Vector3d::operator-(
        const typename open3d::Vector3d &v) const {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = (*this)[c] - v[c];

    return output;
}

typename open3d::Vector3d &open3d::Vector3d::operator+=(
        const typename open3d::Vector3d &v) {
    for (uint c = 0; c < Vector3d::COLS; c++) (*this)[c] += v[c];

    return *this;
}

typename open3d::Vector3d &open3d::Vector3d::operator-=(
        const typename open3d::Vector3d &v) {
    for (uint c = 0; c < Vector3d::COLS; c++) (*this)[c] -= v[c];

    return *this;
}

typename open3d::Vector3d open3d::Vector3d::operator+(const double &t) const {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = (*this)[c] + t;

    return output;
}

typename open3d::Vector3d open3d::Vector3d::operator-(const double &t) const {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = (*this)[c] - t;

    return output;
}

typename open3d::Vector3d open3d::Vector3d::operator*(const double &t) const {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = (*this)[c] * t;

    return output;
}

typename open3d::Vector3d open3d::Vector3d::operator/(const double &t) const {
    Vector3d output;
    for (uint c = 0; c < Vector3d::COLS; c++) output[c] = (*this)[c] / t;

    return output;
}

typename open3d::Vector3d &open3d::Vector3d::operator+=(const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) (*this)[c] += t;

    return *this;
}

typename open3d::Vector3d &open3d::Vector3d::operator-=(const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) (*this)[c] -= t;

    return *this;
}

typename open3d::Vector3d &open3d::Vector3d::operator*=(const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) (*this)[c] *= t;

    return *this;
}

typename open3d::Vector3d &open3d::Vector3d::operator/=(const double &t) {
    for (uint c = 0; c < Vector3d::COLS; c++) (*this)[c] /= t;

    return *this;
}
