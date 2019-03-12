#pragma once

#include "setup.h"

typedef unsigned int uint;

namespace open3d {
// 1D tensor, row major
typedef struct _Vector6f {
    static const uint ROWS = 1;
    static const uint COLS = 6;

    float s[ROWS][COLS];

    OPEN3D_FUNC_DECL inline
    float &operator[](const uint &i) {
        assert(i < COLS);

        return s[0][i];
    }

    OPEN3D_FUNC_DECL inline
    const float &operator[](const uint &i) const {
        assert(i < COLS);

        return s[0][i];
    }
} Vector6f;
}  // namespace open3d
