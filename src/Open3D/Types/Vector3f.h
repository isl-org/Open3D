#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector3f {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    float s[ROWS][COLS];

    float &operator[](const uint &i);
    const float &operator[](const uint &i) const;
} Vector3f;
}  // namespace open3d
