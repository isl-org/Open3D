#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3i {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    int s[COLS];

    int &operator[](const uint &i);
    const int &operator[](const uint &i) const;
    explicit operator int *const();
    explicit operator const int *const();
} Vector3i;

bool operator==(const Vector3i &v0, const Vector3i &v1);
bool operator!=(const Vector3i &v0, const Vector3i &v1);
bool operator<=(const Vector3i &v0, const Vector3i &v1);
bool operator>=(const Vector3i &v0, const Vector3i &v1);

Vector3i operator+(const Vector3i &v0, const Vector3i &v1);
Vector3i operator-(const Vector3i &v0, const Vector3i &v1);
Vector3i &operator+=(Vector3i &v0, const Vector3i &v1);
Vector3i &operator-=(Vector3i &v0, const Vector3i &v1);
Vector3i operator+(const Vector3i &v, const int &t);
Vector3i operator-(const Vector3i &v, const int &t);
Vector3i operator*(const Vector3i &v, const int &t);
Vector3i operator/(const Vector3i &v, const int &t);
Vector3i &operator+=(Vector3i &v, const int &t);
Vector3i &operator-=(Vector3i &v, const int &t);
Vector3i &operator*=(Vector3i &v, const int &t);
Vector3i &operator/=(Vector3i &v, const int &t);
}  // namespace open3d
