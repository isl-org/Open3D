#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6f {
    static const uint ROWS = 6;
    static const uint COLS = 6;

    float s[ROWS][COLS];

    // subscript operator: readwrite
    float* operator[](const uint& i);
    // subscript operator: readonly
    const float* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator float* const();
    // casting operator: readonly
    explicit operator const float* const();
} Matrix6f;

    bool operator==(const Matrix6f& m0, const Matrix6f& m1);
    bool operator!=(const Matrix6f& m0, const Matrix6f& m1);
    bool operator<=(const Matrix6f& m0, const Matrix6f& m1);
    bool operator>=(const Matrix6f& m0, const Matrix6f& m1);

    // addition
    Matrix6f operator+(const Matrix6f& m0, const Matrix6f& m1);
    // subtraction
    Matrix6f operator-(const Matrix6f& m0, const Matrix6f& m1);
    // addition assignment
    Matrix6f& operator+=(Matrix6f& m0, const Matrix6f& m1);
    // subtraction assignment
    Matrix6f& operator-=(Matrix6f& m0, const Matrix6f& m1);
    // addition
    Matrix6f operator+(const Matrix6f& m, const float& t);
    // subtraction
    Matrix6f operator-(const Matrix6f& m, const float& t);
    // multiply with scalar
    Matrix6f operator*(const Matrix6f& m, const float& t);
    // divide by scalar
    Matrix6f operator/(const Matrix6f& m, const float& t);
    // addition assignment
    Matrix6f& operator+=(Matrix6f& m, const float& t);
    // subtraction assignment
    Matrix6f& operator-=(Matrix6f& m, const float& t);
    // multiplication assignment
    Matrix6f& operator*=(Matrix6f& m, const float& t);
    // division assignment
    Matrix6f& operator/=(Matrix6f& m, const float& t);
}  // namespace open3d
