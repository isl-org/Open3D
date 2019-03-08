#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3f {
    static const uint ROWS = 3;
    static const uint COLS = 3;

    float s[ROWS][COLS];

    // subscript operator: readwrite
    float* operator[](const uint& i);
    // subscript operator: readonly
    const float* operator[](const uint& i) const;

    // casting operator: readwrite
    explicit operator float* const();
    // casting operator: readonly
    explicit operator const float* const();

    bool operator==(const _Matrix3f& m);
    bool operator!=(const _Matrix3f& m);
    bool operator<=(const _Matrix3f& m);
    bool operator>=(const _Matrix3f& m);

    // addition
    _Matrix3f operator+(const _Matrix3f& m) const;
    // subtraction
    _Matrix3f operator-(const _Matrix3f& m) const;
    // addition assignment
    _Matrix3f& operator+=(const _Matrix3f& m);
    // subtraction assignment
    _Matrix3f& operator-=(const _Matrix3f& m);
    // addition
    _Matrix3f operator+(const float& t) const;
    // subtraction
    _Matrix3f operator-(const float& t) const;
    // multiply with scalar
    _Matrix3f operator*(const float& t) const;
    // divide by scalar
    _Matrix3f operator/(const float& t) const;
    // addition assignment
    _Matrix3f& operator+=(const float& t);
    // subtraction assignment
    _Matrix3f& operator-=(const float& t);
    // multiplication assignment
    _Matrix3f& operator*=(const float& t);
    // division assignment
    _Matrix3f& operator/=(const float& t);
} Matrix3f;
}  // namespace open3d
