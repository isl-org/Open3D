#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3f {
    static const uint ROWS = 3;
    static const uint COLS = 3;

    float s[ROWS][COLS];

    float* operator[](const uint& i);
    const float* operator[](const uint& i) const;
    explicit operator float* const();
    explicit operator const float* const();
} Matrix3f;

Matrix3f operator+(const Matrix3f& m0, const Matrix3f& m1);
Matrix3f operator-(const Matrix3f& m0, const Matrix3f& m1);
Matrix3f& operator+=(Matrix3f& m0, const Matrix3f& m1);
Matrix3f& operator-=(Matrix3f& m0, const Matrix3f& m1);
Matrix3f operator+(const Matrix3f& m, const float& t);
Matrix3f operator-(const Matrix3f& m, const float& t);
Matrix3f operator*(const Matrix3f& m, const float& t);
Matrix3f operator/(const Matrix3f& m, const float& t);
Matrix3f& operator+=(Matrix3f& m, const float& t);
Matrix3f& operator-=(Matrix3f& m, const float& t);
Matrix3f& operator*=(Matrix3f& m, const float& t);
Matrix3f& operator/=(Matrix3f& m, const float& t);
}  // namespace open3d
