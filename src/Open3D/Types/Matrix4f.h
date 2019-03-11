#pragma once

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4f {
    static const uint ROWS = 4;
    static const uint COLS = 4;

    float s[ROWS][COLS];

    float* operator[](const uint& i);
    const float* operator[](const uint& i) const;
} Matrix4f;
}  // namespace open3d
