
#include "Matrix3d.h"

#include <cassert>

double *open3d::Matrix3d::operator[](const uint &i) {
    // catch error in debug mode
    assert(0 <= i && i < Matrix3d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix3d::operator[](const uint &i) const {
    // catch error in debug mode
    assert(0 <= i && i < Matrix3d::ROWS);

    return (const double *const) & s[i];
}

open3d::Matrix3d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Matrix3d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}

bool open3d::operator==(const Matrix3d& m0, const Matrix3d& m1) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            if (m0[r][c] != m1[r][c]) return false;

    return true;
}

bool open3d::operator!=(const Matrix3d& m0, const Matrix3d& m1) {
    return !(m0 == m1);
}

bool open3d::operator<=(const Matrix3d& m0, const Matrix3d& m1) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            if (m0[r][c] > m1[r][c]) return false;

    return true;
}

bool open3d::operator>=(const Matrix3d& m0, const Matrix3d& m1) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            if (m0[r][c] < m1[r][c]) return false;

    return true;
}

open3d::Matrix3d open3d::operator+(const Matrix3d &m0, const Matrix3d &m1) {
    Matrix3d output;
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            output[r][c] = m0[r][c] + m1[r][c];

    return output;
}

open3d::Matrix3d open3d::operator-(const Matrix3d &m0, const Matrix3d &m1) {
    Matrix3d output;
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            output[r][c] = m0[r][c] - m1[r][c];

    return output;
}

open3d::Matrix3d &open3d::operator+=(Matrix3d &m0, const Matrix3d &m1) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) m0[r][c] += m1[r][c];

    return m0;
}

open3d::Matrix3d &open3d::operator-=(Matrix3d &m0, const Matrix3d &m1) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) m0[r][c] -= m1[r][c];

    return m0;
}

open3d::Matrix3d open3d::operator+(const open3d::Matrix3d &m, const double &t) {
    Matrix3d output;
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) output[r][c] = m[r][c] + t;

    return output;
}

open3d::Matrix3d open3d::operator-(const open3d::Matrix3d &m, const double &t) {
    Matrix3d output;
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) output[r][c] = m[r][c] - t;

    return output;
}

open3d::Matrix3d open3d::operator*(const open3d::Matrix3d &m, const double &t) {
    Matrix3d output;
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) output[r][c] = m[r][c] * t;

    return output;
}

open3d::Matrix3d open3d::operator/(const open3d::Matrix3d &m, const double &t) {
    Matrix3d output;
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) output[r][c] = m[r][c] / t;

    return output;
}

open3d::Matrix3d &open3d::operator+=(open3d::Matrix3d &m, const double &t) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) m[r][c] += t;

    return m;
}

open3d::Matrix3d &open3d::operator-=(open3d::Matrix3d &m, const double &t) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) m[r][c] -= t;

    return m;
}

open3d::Matrix3d &open3d::operator*=(open3d::Matrix3d &m, const double &t) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) m[r][c] *= t;

    return m;
}

open3d::Matrix3d &open3d::operator/=(open3d::Matrix3d &m, const double &t) {
    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++) m[r][c] /= t;

    return m;
}
