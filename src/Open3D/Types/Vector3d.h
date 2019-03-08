#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct _Vector3d {
    static const uint COLS = 3;

    double s[COLS];

    // subscript operator: readwrite
    double &operator[](const uint &i);
    // subscript operator: readonly
    const double &operator[](const uint &i) const;

    // casting operator: readwrite
    explicit operator double *const();
    // casting operator: readonly
    explicit operator const double *const();

    bool operator==(const _Vector3d &m);
    bool operator!=(const _Vector3d &m);
    bool operator<=(const _Vector3d &m);
    bool operator>=(const _Vector3d &m);

    // addition
    _Vector3d operator+(const _Vector3d &v) const;
    // subtraction
    _Vector3d operator-(const _Vector3d &v) const;
    // addition assignment
    _Vector3d &operator+=(const _Vector3d &v);
    // subtraction assignment
    _Vector3d &operator-=(const _Vector3d &v);
    // addition
    _Vector3d operator+(const double &t) const;
    // subtraction
    _Vector3d operator-(const double &t) const;
    // multiply with scalar
    _Vector3d operator*(const double &t) const;
    // divide by scalar
    _Vector3d operator/(const double &t) const;
    // addition assignment
    _Vector3d &operator+=(const double &t);
    // subtraction assignment
    _Vector3d &operator-=(const double &t);
    // multiplication assignment
    _Vector3d &operator*=(const double &t);
    // division assignment
    _Vector3d &operator/=(const double &t);
} Vector3d;
}  // namespace open3d
