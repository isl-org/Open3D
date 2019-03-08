
#include "Matrix4d.h"

#include <cassert>

double *open3d::Matrix4d::operator[](const uint &i) {
    // catch error in debug mode
    assert(0 <= i && i < Matrix4d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix4d::operator[](const uint &i) const {
    // catch error in debug mode
    assert(0 <= i && i < Matrix4d::ROWS);

    return (const double *const) & s[i];
}

open3d::Matrix4d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Matrix4d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}

bool open3d::Matrix4d::operator==(const typename open3d::Matrix4d &m) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            if ((*this)[r][c] != m[r][c]) return false;

    return true;
}

bool open3d::Matrix4d::operator!=(const typename open3d::Matrix4d &m) {
    return !(*this == m);
}

bool open3d::Matrix4d::operator<=(const typename open3d::Matrix4d &m) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            if ((*this)[r][c] > m[r][c]) return false;

    return true;
}

bool open3d::Matrix4d::operator>=(const typename open3d::Matrix4d &m) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        if ((*this)[r] < m[r]) return false;

    return true;
}

typename open3d::Matrix4d open3d::Matrix4d::operator+(
        const typename open3d::Matrix4d &m) const {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = (*this)[r][c] + m[r][c];

    return output;
}

typename open3d::Matrix4d open3d::Matrix4d::operator-(
        const typename open3d::Matrix4d &m) const {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = (*this)[r][c] - m[r][c];

    return output;
}

typename open3d::Matrix4d &open3d::Matrix4d::operator+=(
        const typename open3d::Matrix4d &m) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) (*this)[r][c] += m[r][c];

    return *this;
}

typename open3d::Matrix4d &open3d::Matrix4d::operator-=(
        const typename open3d::Matrix4d &m) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) (*this)[r][c] -= m[r][c];

    return *this;
}

typename open3d::Matrix4d open3d::Matrix4d::operator+(const double &t) const {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = (*this)[r][c] + t;

    return output;
}

typename open3d::Matrix4d open3d::Matrix4d::operator-(const double &t) const {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = (*this)[r][c] - t;

    return output;
}

typename open3d::Matrix4d open3d::Matrix4d::operator*(const double &t) const {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = (*this)[r][c] * t;

    return output;
}

typename open3d::Matrix4d open3d::Matrix4d::operator/(const double &t) const {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = (*this)[r][c] / t;

    return output;
}

typename open3d::Matrix4d &open3d::Matrix4d::operator+=(const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) (*this)[r][c] += t;

    return *this;
}

typename open3d::Matrix4d &open3d::Matrix4d::operator-=(const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) (*this)[r][c] -= t;

    return *this;
}

typename open3d::Matrix4d &open3d::Matrix4d::operator*=(const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) (*this)[r][c] *= t;

    return *this;
}

typename open3d::Matrix4d &open3d::Matrix4d::operator/=(const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) (*this)[r][c] /= t;

    return *this;
}
