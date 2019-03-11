#pragma once

#include "Vector6d.h"

typedef unsigned int uint;

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6d {
    static const uint ROWS = 6;
    static const uint COLS = Vector6d::COLS;

    Vector6d s[ROWS];

    Vector6d& operator[](const uint& i);
    const Vector6d& operator[](const uint& i) const;
} Matrix6d;
}  // namespace open3d
