#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6f {
    static const uint ROWS = 6;
    static const uint COLS = 6;

    float s[ROWS][COLS];

    float* operator[](const uint& i);
    const float* operator[](const uint& i) const;
    explicit operator float* const();
    explicit operator const float* const();
} Matrix6f;

Matrix6f operator+(const Matrix6f& m0, const Matrix6f& m1);
Matrix6f operator-(const Matrix6f& m0, const Matrix6f& m1);
Matrix6f& operator+=(Matrix6f& m0, const Matrix6f& m1);
Matrix6f& operator-=(Matrix6f& m0, const Matrix6f& m1);
Matrix6f operator+(const Matrix6f& m, const float& t);
Matrix6f operator-(const Matrix6f& m, const float& t);
Matrix6f operator*(const Matrix6f& m, const float& t);
Matrix6f operator/(const Matrix6f& m, const float& t);
Matrix6f& operator+=(Matrix6f& m, const float& t);
Matrix6f& operator-=(Matrix6f& m, const float& t);
Matrix6f& operator*=(Matrix6f& m, const float& t);
Matrix6f& operator/=(Matrix6f& m, const float& t);
}  // namespace open3d
