#pragma once

#include "Vector4d.h"

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4d {
    static const uint ROWS = 4;
    static const uint COLS = Vector4d::COLS;

    Vector4d s[ROWS];

    OPEN3D_FUNC_DECL inline
    Vector4d& operator[](const uint &i) {
        assert(i < ROWS);

        return s[i];
    }

    OPEN3D_FUNC_DECL inline
    const Vector4d& operator[](const uint &i) const {
        assert(i < ROWS);

        return s[i];
    }
} Matrix4d;
}  // namespace open3d
