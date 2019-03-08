#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3d {
    static const uint ROWS = 1;
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
} Vector3d;

    bool operator==(const Vector3d &v0, const Vector3d &v1);
    bool operator!=(const Vector3d &v0, const Vector3d &v1);
    bool operator<=(const Vector3d &v0, const Vector3d &v1);
    bool operator>=(const Vector3d &v0, const Vector3d &v1);

    // addition
    Vector3d operator+(const Vector3d &v0, const Vector3d &v1);
    // subtraction
    Vector3d operator-(const Vector3d &v0, const Vector3d &v1);
    // addition assignment
    Vector3d &operator+=(Vector3d &v0, const Vector3d &v1);
    // subtraction assignment
    Vector3d &operator-=(Vector3d &v0, const Vector3d &v1);
    // addition
    Vector3d operator+(const Vector3d &v, const double &t);
    // subtraction
    Vector3d operator-(const Vector3d &v, const double &t);
    // multiply with scalar
    Vector3d operator*(const Vector3d &v, const double &t);
    // divide by scalar
    Vector3d operator/(const Vector3d &v, const double &t);
    // addition assignment
    Vector3d &operator+=(Vector3d &v, const double &t);
    // subtraction assignment
    Vector3d &operator-=(Vector3d &v, const double &t);
    // multiplication assignment
    Vector3d &operator*=(Vector3d &v, const double &t);
    // division assignment
    Vector3d &operator/=(Vector3d &v, const double &t);
}  // namespace open3d
