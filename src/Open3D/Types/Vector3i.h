#pragma once

#include "setup.h"

namespace open3d {
// 1D tensor, row major
typedef struct _Vector3i {
    static const uint ROWS = 1;
    static const uint COLS = 3;

    int s[ROWS][COLS];

    OPEN3D_FUNC_DECL inline
    int &operator[](const uint &i) {
        assert(i < COLS);

        return s[0][i];
    }

    OPEN3D_FUNC_DECL inline
    const int &operator[](const uint &i) const {
        assert(i < COLS);

        return s[0][i];
    }
} Vector3i;
}  // namespace open3d
