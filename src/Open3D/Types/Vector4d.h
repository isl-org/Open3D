#pragma once

#include "setup.h"

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct _Vector4d {
    static const uint ROWS = 1;
    static const uint COLS = 4;

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
} Vector4d;
}  // namespace open3d
