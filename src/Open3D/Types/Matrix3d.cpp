
#include "Matrix3d.h"
using namespace open3d;

#include <cassert>

Vector3d &open3d::Matrix3d::operator[](const uint &i) {
    assert(i < Matrix3d::ROWS);

    return s[i];
}

const Vector3d &open3d::Matrix3d::operator[](const uint &i) const {
    assert(i < Matrix3d::ROWS);

    return s[i];
}
