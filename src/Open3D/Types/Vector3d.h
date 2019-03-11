#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3d {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    double s[ROWS][COLS];

    double &operator[](const uint &i);
    const double &operator[](const uint &i) const;
} Vector3d;
}  // namespace open3d
