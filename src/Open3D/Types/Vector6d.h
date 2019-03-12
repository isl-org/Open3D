#pragma once

#include "setup.h"

namespace open3d {
// 1D tensor, row major
typedef struct _Vector6d {
    static const uint ROWS = 1;
    static const uint COLS = 6;

    double s[ROWS][COLS];

    OPEN3D_FUNC_DECL inline
    double &operator[](const uint &i) {
        assert(i < COLS);

        return s[0][i];
    }

    OPEN3D_FUNC_DECL inline
    const double &operator[](const uint &i) const {
        assert(i < COLS);

        return s[0][i];
    }
} Vector6d;
}  // namespace open3d
