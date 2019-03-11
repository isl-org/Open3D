#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector6d {
    static const uint ROWS = 1;
    static const uint COLS = 6;

    double s[ROWS][COLS];

    double &operator[](const uint &i);
    const double &operator[](const uint &i) const;
} Vector6d;
}  // namespace open3d
