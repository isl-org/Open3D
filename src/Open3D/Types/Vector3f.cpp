
#include "Vector3f.h"

#include <cassert>

float &open3d::Vector3f::operator[](const uint &i) {
    assert(i < Vector3f::COLS);

    return s[0][i];
}

const float &open3d::Vector3f::operator[](const uint &i) const {
    assert(i < Vector3f::COLS);

    return s[0][i];
}
