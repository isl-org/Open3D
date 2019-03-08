#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct _Vector3f {
    static const uint COLS = 3;

    float s[COLS];

    // subscript operator: readwrite
    float &operator[](const uint &i);
    // subscript operator: readonly
    const float &operator[](const uint &i) const;

    // casting operator: readwrite
    explicit operator float *const();
    // casting operator: readonly
    explicit operator const float *const();

    bool operator==(const _Vector3f &m);
    bool operator!=(const _Vector3f &m);
    bool operator<=(const _Vector3f &m);
    bool operator>=(const _Vector3f &m);

    // addition
    _Vector3f operator+(const _Vector3f &v) const;
    // subtraction
    _Vector3f operator-(const _Vector3f &v) const;
    // addition assignment
    _Vector3f &operator+=(const _Vector3f &v);
    // subtraction assignment
    _Vector3f &operator-=(const _Vector3f &v);
    // addition
    _Vector3f operator+(const float &t) const;
    // subtraction
    _Vector3f operator-(const float &t) const;
    // multiply with scalar
    _Vector3f operator*(const float &t) const;
    // divide by scalar
    _Vector3f operator/(const float &t) const;
    // addition assignment
    _Vector3f &operator+=(const float &t);
    // subtraction assignment
    _Vector3f &operator-=(const float &t);
    // multiplication assignment
    _Vector3f &operator*=(const float &t);
    // division assignment
    _Vector3f &operator/=(const float &t);
} Vector3f;
}  // namespace open3d
