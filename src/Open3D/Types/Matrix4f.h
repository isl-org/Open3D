#pragma once

#include "Vector4f.h"

namespace open3d {
// 2D tensor, row major
typedef struct _Matrix4f {
    static const uint ROWS = 4;
    static const uint COLS = Vector4f::COLS;

    Vector4f s[ROWS];

    OPEN3D_FUNC_DECL inline
    Vector4f& operator[](const uint &i) {
        assert(i < ROWS);

        return s[i];
    }

    OPEN3D_FUNC_DECL inline
    const Vector4f& operator[](const uint &i) const {
        assert(i < ROWS);

        return s[i];
    }
} Matrix4f;
}  // namespace open3d
