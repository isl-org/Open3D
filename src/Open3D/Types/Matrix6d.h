#pragma once

#include "Vector6d.h"

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6d {
    static const uint ROWS = 6;
    static const uint COLS = Vector6d::COLS;

    Vector6d s[ROWS];

    OPEN3D_FUNC_DECL inline
    Vector6d& operator[](const uint &i) {
        assert(i < ROWS);

        return s[i];
    }

    OPEN3D_FUNC_DECL inline
    const Vector6d& operator[](const uint &i) const {
        assert(i < ROWS);

        return s[i];
    }
} Matrix6d;
}  // namespace open3d
