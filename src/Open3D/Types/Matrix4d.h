#pragma once

#include "Vector4d.h"

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4d {
    static const uint ROWS = 4;
    static const uint COLS = Vector4d::COLS;

    Vector4d s[ROWS];

    Vector4d& operator[](const uint& i);
    const Vector4d& operator[](const uint& i) const;
} Matrix4d;
}  // namespace open3d
