#pragma once

#include "Vector3f.h"

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3f {
    static const uint ROWS = 3;
    static const uint COLS = Vector3f::COLS;

    Vector3f s[ROWS];

    Vector3f& operator[](const uint& i);
    const Vector3f& operator[](const uint& i) const;
} Matrix3f;
}  // namespace open3d
