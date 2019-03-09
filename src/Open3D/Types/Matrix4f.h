#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4f {
    static const uint ROWS = 4;
    static const uint COLS = 4;

    float s[ROWS][COLS];

    float* operator[](const uint& i);
    const float* operator[](const uint& i) const;
    explicit operator float* const();
    explicit operator const float* const();
} Matrix4f;

Matrix4f operator+(const Matrix4f& m0, const Matrix4f& m1);
Matrix4f operator-(const Matrix4f& m0, const Matrix4f& m1);
Matrix4f& operator+=(Matrix4f& m0, const Matrix4f& m1);
Matrix4f& operator-=(Matrix4f& m0, const Matrix4f& m1);
Matrix4f operator+(const Matrix4f& m, const float& t);
Matrix4f operator-(const Matrix4f& m, const float& t);
Matrix4f operator*(const Matrix4f& m, const float& t);
Matrix4f operator/(const Matrix4f& m, const float& t);
Matrix4f& operator+=(Matrix4f& m, const float& t);
Matrix4f& operator-=(Matrix4f& m, const float& t);
Matrix4f& operator*=(Matrix4f& m, const float& t);
Matrix4f& operator/=(Matrix4f& m, const float& t);
}  // namespace open3d
