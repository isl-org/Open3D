#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6f {
    static const uint ROWS = 6;
    static const uint COLS = 6;

    float s[ROWS][COLS];

    float* operator[](const uint& i);
    const float* operator[](const uint& i) const;
} Matrix6f;
}  // namespace open3d
