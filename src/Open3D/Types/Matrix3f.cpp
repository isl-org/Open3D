
#include "Matrix3f.h"
using namespace open3d;

#include <cassert>

Vector3f &open3d::Matrix3f::operator[](const uint &i) {
    assert(i < Matrix3f::ROWS);

    return s[i];
}

const Vector3f &open3d::Matrix3f::operator[](const uint &i) const {
    assert(i < Matrix3f::ROWS);

    return s[i];
}
