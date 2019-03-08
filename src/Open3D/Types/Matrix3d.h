#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3d {
    static const uint ROWS = 3;
    static const uint COLS = 3;

    double s[ROWS][COLS];

    double* operator[](const uint& i);
    const double* operator[](const uint& i) const;
    explicit operator double* const();
    explicit operator const double* const();
} Matrix3d;

bool operator==(const Matrix3d& m0, const Matrix3d& m1);
bool operator!=(const Matrix3d& m0, const Matrix3d& m1);
bool operator<=(const Matrix3d& m0, const Matrix3d& m1);
bool operator>=(const Matrix3d& m0, const Matrix3d& m1);

Matrix3d operator+(const Matrix3d& m0, const Matrix3d& m1);
Matrix3d operator-(const Matrix3d& m0, const Matrix3d& m1);
Matrix3d& operator+=(Matrix3d& m0, const Matrix3d& m1);
Matrix3d& operator-=(Matrix3d& m0, const Matrix3d& m1);
Matrix3d operator+(const Matrix3d& m, const double& t);
Matrix3d operator-(const Matrix3d& m, const double& t);
Matrix3d operator*(const Matrix3d& m, const double& t);
Matrix3d operator/(const Matrix3d& m, const double& t);
Matrix3d& operator+=(Matrix3d& m, const double& t);
Matrix3d& operator-=(Matrix3d& m, const double& t);
Matrix3d& operator*=(Matrix3d& m, const double& t);
Matrix3d& operator/=(Matrix3d& m, const double& t);
}  // namespace open3d
