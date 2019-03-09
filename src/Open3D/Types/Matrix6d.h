#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6d {
    static const uint ROWS = 6;
    static const uint COLS = 6;

    double s[ROWS][COLS];

    double* operator[](const uint& i);
    const double* operator[](const uint& i) const;
    explicit operator double* const();
    explicit operator const double* const();
} Matrix6d;

Matrix6d operator+(const Matrix6d& m0, const Matrix6d& m1);
Matrix6d operator-(const Matrix6d& m0, const Matrix6d& m1);
Matrix6d& operator+=(Matrix6d& m0, const Matrix6d& m1);
Matrix6d& operator-=(Matrix6d& m0, const Matrix6d& m1);
Matrix6d operator+(const Matrix6d& m, const double& t);
Matrix6d operator-(const Matrix6d& m, const double& t);
Matrix6d operator*(const Matrix6d& m, const double& t);
Matrix6d operator/(const Matrix6d& m, const double& t);
Matrix6d& operator+=(Matrix6d& m, const double& t);
Matrix6d& operator-=(Matrix6d& m, const double& t);
Matrix6d& operator*=(Matrix6d& m, const double& t);
Matrix6d& operator/=(Matrix6d& m, const double& t);
}  // namespace open3d
