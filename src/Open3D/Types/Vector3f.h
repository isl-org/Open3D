#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3f {
    static const uint ROWS = 1;
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
} Vector3f;

bool operator==(const Vector3f &v0, const Vector3f &v1);
bool operator!=(const Vector3f &v0, const Vector3f &v1);
bool operator<=(const Vector3f &v0, const Vector3f &v1);
bool operator>=(const Vector3f &v0, const Vector3f &v1);

// addition
Vector3f operator+(const Vector3f &v0, const Vector3f &v1);
// subtraction
Vector3f operator-(const Vector3f &v0, const Vector3f &v1);
// addition assignment
Vector3f &operator+=(Vector3f &v0, const Vector3f &v1);
// subtraction assignment
Vector3f &operator-=(Vector3f &v0, const Vector3f &v1);
// addition
Vector3f operator+(const Vector3f &v, const float &t);
// subtraction
Vector3f operator-(const Vector3f &v, const float &t);
// multiply with scalar
Vector3f operator*(const Vector3f &v, const float &t);
// divide by scalar
Vector3f operator/(const Vector3f &v, const float &t);
// addition assignment
Vector3f &operator+=(Vector3f &v, const float &t);
// subtraction assignment
Vector3f &operator-=(Vector3f &v, const float &t);
// multiplication assignment
Vector3f &operator*=(Vector3f &v, const float &t);
// division assignment
Vector3f &operator/=(Vector3f &v, const float &t);
}  // namespace open3d
