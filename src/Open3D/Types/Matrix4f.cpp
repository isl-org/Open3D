
#include "Matrix4f.h"
using namespace open3d;

#include <cassert>

Vector4f& open3d::Matrix4f::operator[](const uint &i) {
    assert(i < Matrix4f::ROWS);

    return s[i];
}

const Vector4f& open3d::Matrix4f::operator[](const uint &i) const {
    assert(i < Matrix4f::ROWS);

    return s[i];
}
