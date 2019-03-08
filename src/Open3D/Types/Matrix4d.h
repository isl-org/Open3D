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

    bool operator==(const _Matrix4d& m);
    bool operator!=(const _Matrix4d& m);
    bool operator<=(const _Matrix4d& m);
    bool operator>=(const _Matrix4d& m);

    // addition
    _Matrix4d operator+(const _Matrix4d& m) const;
    // subtraction
    _Matrix4d operator-(const _Matrix4d& m) const;
    // addition assignment
    _Matrix4d& operator+=(const _Matrix4d& m);
    // subtraction assignment
    _Matrix4d& operator-=(const _Matrix4d& m);
    // addition
    _Matrix4d operator+(const double& t) const;
    // subtraction
    _Matrix4d operator-(const double& t) const;
    // multiply with scalar
    _Matrix4d operator*(const double& t) const;
    // divide by scalar
    _Matrix4d operator/(const double& t) const;
    // addition assignment
    _Matrix4d& operator+=(const double& t);
    // subtraction assignment
    _Matrix4d& operator-=(const double& t);
    // multiplication assignment
    _Matrix4d& operator*=(const double& t);
    // division assignment
    _Matrix4d& operator/=(const double& t);
} Matrix4d;
}  // namespace open3d
