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

    bool operator==(const _Matrix3d& m);
    bool operator!=(const _Matrix3d& m);
    bool operator<=(const _Matrix3d& m);
    bool operator>=(const _Matrix3d& m);

    // addition
    _Matrix3d operator+(const _Matrix3d& m) const;
    // subtraction
    _Matrix3d operator-(const _Matrix3d& m) const;
    // addition assignment
    _Matrix3d& operator+=(const _Matrix3d& m);
    // subtraction assignment
    _Matrix3d& operator-=(const _Matrix3d& m);
    // addition
    _Matrix3d operator+(const double& t) const;
    // subtraction
    _Matrix3d operator-(const double& t) const;
    // multiply with scalar
    _Matrix3d operator*(const double& t) const;
    // divide by scalar
    _Matrix3d operator/(const double& t) const;
    // addition assignment
    _Matrix3d& operator+=(const double& t);
    // subtraction assignment
    _Matrix3d& operator-=(const double& t);
    // multiplication assignment
    _Matrix3d& operator*=(const double& t);
    // division assignment
    _Matrix3d& operator/=(const double& t);
} Matrix3d;
}  // namespace open3d
