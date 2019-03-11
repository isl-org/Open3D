
#include "Matrix6d.h"
using namespace open3d;

#include <cassert>

Vector6d& open3d::Matrix6d::operator[](const uint &i) {
    assert(i < Matrix6d::ROWS);

    return s[i];
}

const Vector6d& open3d::Matrix6d::operator[](const uint &i) const {
    assert(i < Matrix6d::ROWS);

    return s[i];
}
