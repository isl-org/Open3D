#pragma once

#include "Vector4f.h"

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4f {
    static const uint ROWS = 4;
    static const uint COLS = Vector4f::COLS;

    Vector4f s[ROWS];

    Vector4f& operator[](const uint& i);
    const Vector4f& operator[](const uint& i) const;
} Matrix4f;
}  // namespace open3d
