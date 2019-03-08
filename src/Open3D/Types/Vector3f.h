#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3f {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    float s[COLS];

    float &operator[](const uint &i);
    const float &operator[](const uint &i) const;
    explicit operator float *const();
    explicit operator const float *const();
} Vector3f;

bool operator==(const Vector3f &v0, const Vector3f &v1);
bool operator!=(const Vector3f &v0, const Vector3f &v1);
bool operator<=(const Vector3f &v0, const Vector3f &v1);
bool operator>=(const Vector3f &v0, const Vector3f &v1);

Vector3f operator+(const Vector3f &v0, const Vector3f &v1);
Vector3f operator-(const Vector3f &v0, const Vector3f &v1);
Vector3f &operator+=(Vector3f &v0, const Vector3f &v1);
Vector3f &operator-=(Vector3f &v0, const Vector3f &v1);
Vector3f operator+(const Vector3f &v, const float &t);
Vector3f operator-(const Vector3f &v, const float &t);
Vector3f operator*(const Vector3f &v, const float &t);
Vector3f operator/(const Vector3f &v, const float &t);
Vector3f &operator+=(Vector3f &v, const float &t);
Vector3f &operator-=(Vector3f &v, const float &t);
Vector3f &operator*=(Vector3f &v, const float &t);
Vector3f &operator/=(Vector3f &v, const float &t);
}  // namespace open3d
