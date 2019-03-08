#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct _Vector3i {
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

    bool operator==(const _Vector3i &m);
    bool operator!=(const _Vector3i &m);
    bool operator<=(const _Vector3i &m);
    bool operator>=(const _Vector3i &m);

    // addition
    _Vector3i operator+(const _Vector3i &v) const;
    // subtraction
    _Vector3i operator-(const _Vector3i &v) const;
    // addition assignment
    _Vector3i &operator+=(const _Vector3i &v);
    // subtraction assignment
    _Vector3i &operator-=(const _Vector3i &v);
    // addition
    _Vector3i operator+(const int &t) const;
    // subtraction
    _Vector3i operator-(const int &t) const;
    // multiply with scalar
    _Vector3i operator*(const int &t) const;
    // divide by scalar
    _Vector3i operator/(const int &t) const;
    // addition assignment
    _Vector3i &operator+=(const int &t);
    // subtraction assignment
    _Vector3i &operator-=(const int &t);
    // multiplication assignment
    _Vector3i &operator*=(const int &t);
    // division assignment
    _Vector3i &operator/=(const int &t);
} Vector3i;
}  // namespace open3d
