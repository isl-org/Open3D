#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4d {
    static const uint ROWS = 4;
    static const uint COLS = 4;

    double s[ROWS][COLS];

    double* operator[](const uint& i);
    const double* operator[](const uint& i) const;
    explicit operator double* const();
    explicit operator const double* const();
} Matrix4d;
}  // namespace open3d
