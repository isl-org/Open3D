
#include "Matrix4d.h"

#include <cassert>

double *open3d::Matrix4d::operator[](const uint &i) {
    assert(i < Matrix4d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix4d::operator[](const uint &i) const {
    assert(i < Matrix4d::ROWS);

    return (const double *const) & s[i];
}

open3d::Matrix4d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Matrix4d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}

bool open3d::operator==(const Matrix4d &m0, const Matrix4d &m1) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            if (m0[r][c] != m1[r][c]) return false;

    return true;
}

bool open3d::operator!=(const Matrix4d &m0, const Matrix4d &m1) {
    return !(m0 == m1);
}

bool open3d::operator<=(const Matrix4d &m0, const Matrix4d &m1) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            if (m0[r][c] > m1[r][c]) return false;

    return true;
}

bool open3d::operator>=(const Matrix4d &m0, const Matrix4d &m1) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            if (m0[r][c] < m1[r][c]) return false;

    return true;
}

open3d::Matrix4d open3d::operator+(const Matrix4d &m0, const Matrix4d &m1) {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = m0[r][c] + m1[r][c];

    return output;
}

open3d::Matrix4d open3d::operator-(const Matrix4d &m0, const Matrix4d &m1) {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++)
            output[r][c] = m0[r][c] - m1[r][c];

    return output;
}

open3d::Matrix4d &open3d::operator+=(Matrix4d &m0, const Matrix4d &m1) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) m0[r][c] += m1[r][c];

    return m0;
}

open3d::Matrix4d &open3d::operator-=(Matrix4d &m0, const Matrix4d &m1) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) m0[r][c] -= m1[r][c];

    return m0;
}

open3d::Matrix4d open3d::operator+(const open3d::Matrix4d &m, const double &t) {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) output[r][c] = m[r][c] + t;

    return output;
}

open3d::Matrix4d open3d::operator-(const open3d::Matrix4d &m, const double &t) {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) output[r][c] = m[r][c] - t;

    return output;
}

open3d::Matrix4d open3d::operator*(const open3d::Matrix4d &m, const double &t) {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) output[r][c] = m[r][c] * t;

    return output;
}

open3d::Matrix4d open3d::operator/(const open3d::Matrix4d &m, const double &t) {
    Matrix4d output;
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) output[r][c] = m[r][c] / t;

    return output;
}

open3d::Matrix4d &open3d::operator+=(open3d::Matrix4d &m, const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) m[r][c] += t;

    return m;
}

open3d::Matrix4d &open3d::operator-=(open3d::Matrix4d &m, const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) m[r][c] -= t;

    return m;
}

open3d::Matrix4d &open3d::operator*=(open3d::Matrix4d &m, const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) m[r][c] *= t;

    return m;
}

open3d::Matrix4d &open3d::operator/=(open3d::Matrix4d &m, const double &t) {
    for (uint r = 0; r < Matrix4d::ROWS; r++)
        for (uint c = 0; c < Matrix4d::COLS; c++) m[r][c] /= t;

    return m;
}
