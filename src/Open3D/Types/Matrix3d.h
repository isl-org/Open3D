#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3d {
    static const uint ROWS = 3;
    static const uint COLS = 3;

    double s[ROWS][COLS];

    // subscript operator: readwrite
    double* operator[](const uint& i);
    // subscript operator: readonly
    const double* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator double* const();
    // casting operator: readonly
    explicit operator const double* const();
} Matrix3d;

bool operator==(const Matrix3d& m0, const Matrix3d& m1);
bool operator!=(const Matrix3d& m0, const Matrix3d& m1);
bool operator<=(const Matrix3d& m0, const Matrix3d& m1);
bool operator>=(const Matrix3d& m0, const Matrix3d& m1);

// addition
Matrix3d operator+(const Matrix3d& m0, const Matrix3d& m1);
// subtraction
Matrix3d operator-(const Matrix3d& m0, const Matrix3d& m1);
// addition assignment
Matrix3d& operator+=(Matrix3d& m0, const Matrix3d& m1);
// subtraction assignment
Matrix3d& operator-=(Matrix3d& m0, const Matrix3d& m1);
// addition
Matrix3d operator+(const Matrix3d& m, const double& t);
// subtraction
Matrix3d operator-(const Matrix3d& m, const double& t);
// multiply with scalar
Matrix3d operator*(const Matrix3d& m, const double& t);
// divide by scalar
Matrix3d operator/(const Matrix3d& m, const double& t);
// addition assignment
Matrix3d& operator+=(Matrix3d& m, const double& t);
// subtraction assignment
Matrix3d& operator-=(Matrix3d& m, const double& t);
// multiplication assignment
Matrix3d& operator*=(Matrix3d& m, const double& t);
// division assignment
Matrix3d& operator/=(Matrix3d& m, const double& t);
}  // namespace open3d
