#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3f {
    static const uint ROWS = 3;
    static const uint COLS = 3;

    float s[ROWS][COLS];

    float* operator[](const uint& i);
    const float* operator[](const uint& i) const;
} Matrix3f;
}  // namespace open3d
