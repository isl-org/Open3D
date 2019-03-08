#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4f {
    static const uint ROWS = 4;
    static const uint COLS = 4;

    float s[ROWS][COLS];

    // subscript operator: readwrite
    float* operator[](const uint& i);
    // subscript operator: readonly
    const float* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator float* const();
    // casting operator: readonly
    explicit operator const float* const();
} Matrix4f;

bool operator==(const Matrix4f& m0, const Matrix4f& m1);
bool operator!=(const Matrix4f& m0, const Matrix4f& m1);
bool operator<=(const Matrix4f& m0, const Matrix4f& m1);
bool operator>=(const Matrix4f& m0, const Matrix4f& m1);

// addition
Matrix4f operator+(const Matrix4f& m0, const Matrix4f& m1);
// subtraction
Matrix4f operator-(const Matrix4f& m0, const Matrix4f& m1);
// addition assignment
Matrix4f& operator+=(Matrix4f& m0, const Matrix4f& m1);
// subtraction assignment
Matrix4f& operator-=(Matrix4f& m0, const Matrix4f& m1);
// addition
Matrix4f operator+(const Matrix4f& m, const float& t);
// subtraction
Matrix4f operator-(const Matrix4f& m, const float& t);
// multiply with scalar
Matrix4f operator*(const Matrix4f& m, const float& t);
// divide by scalar
Matrix4f operator/(const Matrix4f& m, const float& t);
// addition assignment
Matrix4f& operator+=(Matrix4f& m, const float& t);
// subtraction assignment
Matrix4f& operator-=(Matrix4f& m, const float& t);
// multiplication assignment
Matrix4f& operator*=(Matrix4f& m, const float& t);
// division assignment
Matrix4f& operator/=(Matrix4f& m, const float& t);
}  // namespace open3d
