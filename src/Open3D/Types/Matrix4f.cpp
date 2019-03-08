
#include "Matrix4f.h"

#include <cassert>

float *open3d::Matrix4f::operator[](const uint &i) {
    assert(i < Matrix4f::ROWS);

    return (float *)&s[i];
}

const float *open3d::Matrix4f::operator[](const uint &i) const {
    assert(i < Matrix4f::ROWS);

    return (const float *const) & s[i];
}

open3d::Matrix4f::operator float *const() {
    return reinterpret_cast<float *>(s);
}

open3d::Matrix4f::operator const float *const() {
    return reinterpret_cast<const float *const>(s);
}

bool open3d::operator==(const Matrix4f &m0, const Matrix4f &m1) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++)
            if (m0[r][c] != m1[r][c]) return false;

    return true;
}

bool open3d::operator!=(const Matrix4f &m0, const Matrix4f &m1) {
    return !(m0 == m1);
}

bool open3d::operator<=(const Matrix4f &m0, const Matrix4f &m1) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++)
            if (m0[r][c] > m1[r][c]) return false;

    return true;
}

bool open3d::operator>=(const Matrix4f &m0, const Matrix4f &m1) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++)
            if (m0[r][c] < m1[r][c]) return false;

    return true;
}

open3d::Matrix4f open3d::operator+(const Matrix4f &m0, const Matrix4f &m1) {
    Matrix4f output;
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++)
            output[r][c] = m0[r][c] + m1[r][c];

    return output;
}

open3d::Matrix4f open3d::operator-(const Matrix4f &m0, const Matrix4f &m1) {
    Matrix4f output;
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++)
            output[r][c] = m0[r][c] - m1[r][c];

    return output;
}

open3d::Matrix4f &open3d::operator+=(Matrix4f &m0, const Matrix4f &m1) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) m0[r][c] += m1[r][c];

    return m0;
}

open3d::Matrix4f &open3d::operator-=(Matrix4f &m0, const Matrix4f &m1) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) m0[r][c] -= m1[r][c];

    return m0;
}

open3d::Matrix4f open3d::operator+(const open3d::Matrix4f &m, const float &t) {
    Matrix4f output;
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) output[r][c] = m[r][c] + t;

    return output;
}

open3d::Matrix4f open3d::operator-(const open3d::Matrix4f &m, const float &t) {
    Matrix4f output;
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) output[r][c] = m[r][c] - t;

    return output;
}

open3d::Matrix4f open3d::operator*(const open3d::Matrix4f &m, const float &t) {
    Matrix4f output;
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) output[r][c] = m[r][c] * t;

    return output;
}

open3d::Matrix4f open3d::operator/(const open3d::Matrix4f &m, const float &t) {
    Matrix4f output;
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) output[r][c] = m[r][c] / t;

    return output;
}

open3d::Matrix4f &open3d::operator+=(open3d::Matrix4f &m, const float &t) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) m[r][c] += t;

    return m;
}

open3d::Matrix4f &open3d::operator-=(open3d::Matrix4f &m, const float &t) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) m[r][c] -= t;

    return m;
}

open3d::Matrix4f &open3d::operator*=(open3d::Matrix4f &m, const float &t) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) m[r][c] *= t;

    return m;
}

open3d::Matrix4f &open3d::operator/=(open3d::Matrix4f &m, const float &t) {
    for (uint r = 0; r < Matrix4f::ROWS; r++)
        for (uint c = 0; c < Matrix4f::COLS; c++) m[r][c] /= t;

    return m;
}
