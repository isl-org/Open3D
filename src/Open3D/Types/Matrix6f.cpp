
#include "Matrix6f.h"
using namespace open3d;

#include <cassert>

Vector6f& open3d::Matrix6f::operator[](const uint &i) {
    assert(i < Matrix6f::ROWS);

    return s[i];
}

const Vector6f& open3d::Matrix6f::operator[](const uint &i) const {
    assert(i < Matrix6f::ROWS);

    return s[i];
}
