#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6d {
    static const uint ROWS = 6;
    static const uint COLS = 6;

    double s[ROWS][COLS];

    // subscript operator: readwrite
    double* operator[](const uint& i);
    // subscript operator: readonly
    const double* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator double* const();
    // casting operator: readonly
    explicit operator const double* const();
} Matrix6d;

    bool operator==(const Matrix6d& m0, const Matrix6d& m1);
    bool operator!=(const Matrix6d& m0, const Matrix6d& m1);
    bool operator<=(const Matrix6d& m0, const Matrix6d& m1);
    bool operator>=(const Matrix6d& m0, const Matrix6d& m1);

    // addition
    Matrix6d operator+(const Matrix6d& m0, const Matrix6d& m1);
    // subtraction
    Matrix6d operator-(const Matrix6d& m0, const Matrix6d& m1);
    // addition assignment
    Matrix6d& operator+=(Matrix6d& m0, const Matrix6d& m1);
    // subtraction assignment
    Matrix6d& operator-=(Matrix6d& m0, const Matrix6d& m1);
    // addition
    Matrix6d operator+(const Matrix6d& m, const double& t);
    // subtraction
    Matrix6d operator-(const Matrix6d& m, const double& t);
    // multiply with scalar
    Matrix6d operator*(const Matrix6d& m, const double& t);
    // divide by scalar
    Matrix6d operator/(const Matrix6d& m, const double& t);
    // addition assignment
    Matrix6d& operator+=(Matrix6d& m, const double& t);
    // subtraction assignment
    Matrix6d& operator-=(Matrix6d& m, const double& t);
    // multiplication assignment
    Matrix6d& operator*=(Matrix6d& m, const double& t);
    // division assignment
    Matrix6d& operator/=(Matrix6d& m, const double& t);
}  // namespace open3d
