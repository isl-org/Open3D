#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4d {
    static const uint ROWS = 4;
    static const uint COLS = 4;

    double s[ROWS][COLS];

    // subscript operator: readwrite
    double* operator[](const uint& i);
    // subscript operator: readonly
    const double* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator double* const();
    // casting operator: readonly
    explicit operator const double* const();
} Matrix4d;

bool operator==(const Matrix4d& m0, const Matrix4d& m1);
bool operator!=(const Matrix4d& m0, const Matrix4d& m1);
bool operator<=(const Matrix4d& m0, const Matrix4d& m1);
bool operator>=(const Matrix4d& m0, const Matrix4d& m1);

// addition
Matrix4d operator+(const Matrix4d& m0, const Matrix4d& m1);
// subtraction
Matrix4d operator-(const Matrix4d& m0, const Matrix4d& m1);
// addition assignment
Matrix4d& operator+=(Matrix4d& m0, const Matrix4d& m1);
// subtraction assignment
Matrix4d& operator-=(Matrix4d& m0, const Matrix4d& m1);
// addition
Matrix4d operator+(const Matrix4d& m, const double& t);
// subtraction
Matrix4d operator-(const Matrix4d& m, const double& t);
// multiply with scalar
Matrix4d operator*(const Matrix4d& m, const double& t);
// divide by scalar
Matrix4d operator/(const Matrix4d& m, const double& t);
// addition assignment
Matrix4d& operator+=(Matrix4d& m, const double& t);
// subtraction assignment
Matrix4d& operator-=(Matrix4d& m, const double& t);
// multiplication assignment
Matrix4d& operator*=(Matrix4d& m, const double& t);
// division assignment
Matrix4d& operator/=(Matrix4d& m, const double& t);
}  // namespace open3d
