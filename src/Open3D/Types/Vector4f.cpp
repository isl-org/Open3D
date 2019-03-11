
#include "Vector4f.h"

#include <cassert>

float &open3d::Vector4f::operator[](const uint &i) {
    assert(i < Vector4f::COLS);

    return s[0][i];
}

const float &open3d::Vector4f::operator[](const uint &i) const {
    assert(i < Vector4f::COLS);

    return s[0][i];
}
