
#include "Vector6f.h"

#include <cassert>

float &open3d::Vector6f::operator[](const uint &i) {
    assert(i < Vector6f::COLS);

    return s[0][i];
}

const float &open3d::Vector6f::operator[](const uint &i) const {
    assert(i < Vector6f::COLS);

    return s[0][i];
}
