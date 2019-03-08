#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3f {
    static const uint ROWS = 3;
    static const uint COLS = 3;

    float s[ROWS][COLS];

    // subscript operator: readwrite
    float* operator[](const uint& i);
    // subscript operator: readonly
    const float* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator float* const();
    // casting operator: readonly
    explicit operator const float* const();
} Matrix3f;

bool operator==(const Matrix3f& m0, const Matrix3f& m1);
bool operator!=(const Matrix3f& m0, const Matrix3f& m1);
bool operator<=(const Matrix3f& m0, const Matrix3f& m1);
bool operator>=(const Matrix3f& m0, const Matrix3f& m1);

// addition
Matrix3f operator+(const Matrix3f& m0, const Matrix3f& m1);
// subtraction
Matrix3f operator-(const Matrix3f& m0, const Matrix3f& m1);
// addition assignment
Matrix3f& operator+=(Matrix3f& m0, const Matrix3f& m1);
// subtraction assignment
Matrix3f& operator-=(Matrix3f& m0, const Matrix3f& m1);
// addition
Matrix3f operator+(const Matrix3f& m, const float& t);
// subtraction
Matrix3f operator-(const Matrix3f& m, const float& t);
// multiply with scalar
Matrix3f operator*(const Matrix3f& m, const float& t);
// divide by scalar
Matrix3f operator/(const Matrix3f& m, const float& t);
// addition assignment
Matrix3f& operator+=(Matrix3f& m, const float& t);
// subtraction assignment
Matrix3f& operator-=(Matrix3f& m, const float& t);
// multiplication assignment
Matrix3f& operator*=(Matrix3f& m, const float& t);
// division assignment
Matrix3f& operator/=(Matrix3f& m, const float& t);
}  // namespace open3d
