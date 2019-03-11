
#include "Matrix4d.h"
using namespace open3d;

#include <cassert>

Vector4d& open3d::Matrix4d::operator[](const uint &i) {
    assert(i < Matrix4d::ROWS);

    return s[i];
}

const Vector4d& open3d::Matrix4d::operator[](const uint &i) const {
    assert(i < Matrix4d::ROWS);

    return s[i];
}
