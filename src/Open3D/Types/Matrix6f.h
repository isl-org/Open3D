#pragma once

#include "Vector6f.h"

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix6f {
    static const uint ROWS = 6;
    static const uint COLS = Vector6f::COLS;

    Vector6f s[ROWS];

    OPEN3D_FUNC_DECL inline
    Vector6f& operator[](const uint &i) {
        assert(i < ROWS);

        return s[i];
    }

    OPEN3D_FUNC_DECL inline
    const Vector6f& operator[](const uint &i) const {
        assert(i < ROWS);

        return s[i];
    }
} Matrix6f;
}  // namespace open3d
