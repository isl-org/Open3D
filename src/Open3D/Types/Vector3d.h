#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3d {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    double s[COLS];

    double &operator[](const uint &i);
    const double &operator[](const uint &i) const;
    explicit operator double *const();
    explicit operator const double *const();
} Vector3d;

bool operator==(const Vector3d &v0, const Vector3d &v1);
bool operator!=(const Vector3d &v0, const Vector3d &v1);
bool operator<=(const Vector3d &v0, const Vector3d &v1);
bool operator>=(const Vector3d &v0, const Vector3d &v1);

Vector3d operator+(const Vector3d &v0, const Vector3d &v1);
Vector3d operator-(const Vector3d &v0, const Vector3d &v1);
Vector3d &operator+=(Vector3d &v0, const Vector3d &v1);
Vector3d &operator-=(Vector3d &v0, const Vector3d &v1);
Vector3d operator+(const Vector3d &v, const double &t);
Vector3d operator-(const Vector3d &v, const double &t);
Vector3d operator*(const Vector3d &v, const double &t);
Vector3d operator/(const Vector3d &v, const double &t);
Vector3d &operator+=(Vector3d &v, const double &t);
Vector3d &operator-=(Vector3d &v, const double &t);
Vector3d &operator*=(Vector3d &v, const double &t);
Vector3d &operator/=(Vector3d &v, const double &t);
}  // namespace open3d
