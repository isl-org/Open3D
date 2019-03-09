
#include "Matrix6d.h"

#include <cassert>

double *open3d::Matrix6d::operator[](const uint &i) {
    assert(i < Matrix6d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix6d::operator[](const uint &i) const {
    assert(i < Matrix6d::ROWS);

    return (const double *const) & s[i];
}

open3d::Matrix6d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Matrix6d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}

open3d::Matrix6d open3d::operator+(const Matrix6d &m0, const Matrix6d &m1) {
    Matrix6d output;
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++)
            output[r][c] = m0[r][c] + m1[r][c];

    return output;
}

open3d::Matrix6d open3d::operator-(const Matrix6d &m0, const Matrix6d &m1) {
    Matrix6d output;
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++)
            output[r][c] = m0[r][c] - m1[r][c];

    return output;
}

open3d::Matrix6d &open3d::operator+=(Matrix6d &m0, const Matrix6d &m1) {
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) m0[r][c] += m1[r][c];

    return m0;
}

open3d::Matrix6d &open3d::operator-=(Matrix6d &m0, const Matrix6d &m1) {
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) m0[r][c] -= m1[r][c];

    return m0;
}

open3d::Matrix6d open3d::operator+(const open3d::Matrix6d &m, const double &t) {
    Matrix6d output;
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) output[r][c] = m[r][c] + t;

    return output;
}

open3d::Matrix6d open3d::operator-(const open3d::Matrix6d &m, const double &t) {
    Matrix6d output;
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) output[r][c] = m[r][c] - t;

    return output;
}

open3d::Matrix6d open3d::operator*(const open3d::Matrix6d &m, const double &t) {
    Matrix6d output;
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) output[r][c] = m[r][c] * t;

    return output;
}

open3d::Matrix6d open3d::operator/(const open3d::Matrix6d &m, const double &t) {
    Matrix6d output;
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) output[r][c] = m[r][c] / t;

    return output;
}

open3d::Matrix6d &open3d::operator+=(open3d::Matrix6d &m, const double &t) {
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) m[r][c] += t;

    return m;
}

open3d::Matrix6d &open3d::operator-=(open3d::Matrix6d &m, const double &t) {
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) m[r][c] -= t;

    return m;
}

open3d::Matrix6d &open3d::operator*=(open3d::Matrix6d &m, const double &t) {
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) m[r][c] *= t;

    return m;
}

open3d::Matrix6d &open3d::operator/=(open3d::Matrix6d &m, const double &t) {
    for (uint r = 0; r < Matrix6d::ROWS; r++)
        for (uint c = 0; c < Matrix6d::COLS; c++) m[r][c] /= t;

    return m;
}
