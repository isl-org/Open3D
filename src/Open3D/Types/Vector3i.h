#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3i {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    int s[ROWS][COLS];

    int &operator[](const uint &i);
    const int &operator[](const uint &i) const;
} Vector3i;
}  // namespace open3d
