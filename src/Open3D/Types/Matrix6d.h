#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6d {
    static const uint ROWS = 6;
    static const uint COLS = 6;

    double s[ROWS][COLS];

    double* operator[](const uint& i);
    const double* operator[](const uint& i) const;
} Matrix6d;
}  // namespace open3d
