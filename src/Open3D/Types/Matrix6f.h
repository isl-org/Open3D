#pragma once

#include "Vector6f.h"

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6f {
    static const uint ROWS = 6;
    static const uint COLS = Vector6f::COLS;

    Vector6f s[ROWS];

    Vector6f& operator[](const uint& i);
    const Vector6f& operator[](const uint& i) const;
} Matrix6f;
}  // namespace open3d
