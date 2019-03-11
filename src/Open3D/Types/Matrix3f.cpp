
#include "Matrix3f.h"

#include <cassert>

float *open3d::Matrix3f::operator[](const uint &i) {
    assert(i < Matrix3f::ROWS);

    return (float *)&s[i];
}

const float *open3d::Matrix3f::operator[](const uint &i) const {
    assert(i < Matrix3f::ROWS);

    return (const float *const) & s[i];
}
