#pragma once

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct Vector4f {
    static const uint ROWS = 1;
    static const uint COLS = 4;

    float s[ROWS][COLS];

    float &operator[](const uint &i);
    const float &operator[](const uint &i) const;
} Vector4f;
}  // namespace open3d
