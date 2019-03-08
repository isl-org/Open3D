#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3i {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    int s[COLS];

    // subscript operator: readwrite
    int &operator[](const uint &i);
    // subscript operator: readonly
    const int &operator[](const uint &i) const;

    // casting operator: readwrite
    explicit operator int *const();
    // casting operator: readonly
    explicit operator const int *const();
} Vector3i;

    bool operator==(const Vector3i &v0, const Vector3i &v1);
    bool operator!=(const Vector3i &v0, const Vector3i &v1);
    bool operator<=(const Vector3i &v0, const Vector3i &v1);
    bool operator>=(const Vector3i &v0, const Vector3i &v1);

    // addition
    Vector3i operator+(const Vector3i &v0, const Vector3i &v1);
    // subtraction
    Vector3i operator-(const Vector3i &v0, const Vector3i &v1);
    // addition assignment
    Vector3i &operator+=(Vector3i &v0, const Vector3i &v1);
    // subtraction assignment
    Vector3i &operator-=(Vector3i &v0, const Vector3i &v1);
    // addition
    Vector3i operator+(const Vector3i &v, const int &t);
    // subtraction
    Vector3i operator-(const Vector3i &v, const int &t);
    // multiply with scalar
    Vector3i operator*(const Vector3i &v, const int &t);
    // divide by scalar
    Vector3i operator/(const Vector3i &v, const int &t);
    // addition assignment
    Vector3i &operator+=(Vector3i &v, const int &t);
    // subtraction assignment
    Vector3i &operator-=(Vector3i &v, const int &t);
    // multiplication assignment
    Vector3i &operator*=(Vector3i &v, const int &t);
    // division assignment
    Vector3i &operator/=(Vector3i &v, const int &t);
}  // namespace open3d
