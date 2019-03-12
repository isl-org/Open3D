#pragma once

#include "Vector3f.h"

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix3f {
    static const uint ROWS = 3;
    static const uint COLS = Vector3f::COLS;

    Vector3f s[ROWS];

    OPEN3D_FUNC_DECL inline
    Vector3f &operator[](const uint &i) {
        assert(i < ROWS);

        return s[i];
    }

    OPEN3D_FUNC_DECL inline
    const Vector3f &operator[](const uint &i) const {
        assert(i < ROWS);

        return s[i];
    }
} Matrix3f;
}  // namespace open3d
