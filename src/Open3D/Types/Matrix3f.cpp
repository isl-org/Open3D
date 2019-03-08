
#include "Matrix3f.h"

#include <cassert>

float *open3d::Matrix3f::operator[](const uint &i) {
    // catch error in debug mode
    assert(0 <= i && i < Matrix3f::ROWS);

    return (float *)&s[i];
}

const float *open3d::Matrix3f::operator[](const uint &i) const {
    // catch error in debug mode
    assert(0 <= i && i < Matrix3f::ROWS);

    return (const float *const) & s[i];
}

open3d::Matrix3f::operator float *const() {
    return reinterpret_cast<float *>(s);
}

open3d::Matrix3f::operator const float *const() {
    return reinterpret_cast<const float *const>(s);
}

bool open3d::Matrix3f::operator==(const typename open3d::Matrix3f &m) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            if ((*this)[r][c] != m[r][c]) return false;

    return true;
}

bool open3d::Matrix3f::operator!=(const typename open3d::Matrix3f &m) {
    return !(*this == m);
}

bool open3d::Matrix3f::operator<=(const typename open3d::Matrix3f &m) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            if ((*this)[r][c] > m[r][c]) return false;

    return true;
}

bool open3d::Matrix3f::operator>=(const typename open3d::Matrix3f &m) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        if ((*this)[r] < m[r]) return false;

    return true;
}

typename open3d::Matrix3f open3d::Matrix3f::operator+(
        const typename open3d::Matrix3f &m) const {
    Matrix3f output;
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            output[r][c] = (*this)[r][c] + m[r][c];

    return output;
}

typename open3d::Matrix3f open3d::Matrix3f::operator-(
        const typename open3d::Matrix3f &m) const {
    Matrix3f output;
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            output[r][c] = (*this)[r][c] - m[r][c];

    return output;
}

typename open3d::Matrix3f &open3d::Matrix3f::operator+=(
        const typename open3d::Matrix3f &m) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++) (*this)[r][c] += m[r][c];

    return *this;
}

typename open3d::Matrix3f &open3d::Matrix3f::operator-=(
        const typename open3d::Matrix3f &m) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++) (*this)[r][c] -= m[r][c];

    return *this;
}

typename open3d::Matrix3f open3d::Matrix3f::operator+(const float &t) const {
    Matrix3f output;
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            output[r][c] = (*this)[r][c] + t;

    return output;
}

typename open3d::Matrix3f open3d::Matrix3f::operator-(const float &t) const {
    Matrix3f output;
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            output[r][c] = (*this)[r][c] - t;

    return output;
}

typename open3d::Matrix3f open3d::Matrix3f::operator*(const float &t) const {
    Matrix3f output;
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            output[r][c] = (*this)[r][c] * t;

    return output;
}

typename open3d::Matrix3f open3d::Matrix3f::operator/(const float &t) const {
    Matrix3f output;
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++)
            output[r][c] = (*this)[r][c] / t;

    return output;
}

typename open3d::Matrix3f &open3d::Matrix3f::operator+=(const float &t) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++) (*this)[r][c] += t;

    return *this;
}

typename open3d::Matrix3f &open3d::Matrix3f::operator-=(const float &t) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++) (*this)[r][c] -= t;

    return *this;
}

typename open3d::Matrix3f &open3d::Matrix3f::operator*=(const float &t) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++) (*this)[r][c] *= t;

    return *this;
}

typename open3d::Matrix3f &open3d::Matrix3f::operator/=(const float &t) {
    for (uint r = 0; r < Matrix3f::ROWS; r++)
        for (uint c = 0; c < Matrix3f::COLS; c++) (*this)[r][c] /= t;

    return *this;
}
