#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector4d {
    static const uint ROWS = 1;
    static const uint COLS = 4;

    double s[ROWS][COLS];

    double &operator[](const uint &i);
    const double &operator[](const uint &i) const;
} Vector4d;
}  // namespace open3d
