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

    bool operator==(const _Matrix6d& m);
    bool operator!=(const _Matrix6d& m);
    bool operator<=(const _Matrix6d& m);
    bool operator>=(const _Matrix6d& m);

    // addition
    _Matrix6d operator+(const _Matrix6d& m) const;
    // subtraction
    _Matrix6d operator-(const _Matrix6d& m) const;
    // addition assignment
    _Matrix6d& operator+=(const _Matrix6d& m);
    // subtraction assignment
    _Matrix6d& operator-=(const _Matrix6d& m);
    // addition
    _Matrix6d operator+(const double& t) const;
    // subtraction
    _Matrix6d operator-(const double& t) const;
    // multiply with scalar
    _Matrix6d operator*(const double& t) const;
    // divide by scalar
    _Matrix6d operator/(const double& t) const;
    // addition assignment
    _Matrix6d& operator+=(const double& t);
    // subtraction assignment
    _Matrix6d& operator-=(const double& t);
    // multiplication assignment
    _Matrix6d& operator*=(const double& t);
    // division assignment
    _Matrix6d& operator/=(const double& t);
} Matrix6d;
}  // namespace open3d
