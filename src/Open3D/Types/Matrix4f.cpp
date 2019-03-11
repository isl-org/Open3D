
#include "Matrix4f.h"

#include <cassert>

float *open3d::Matrix4f::operator[](const uint &i) {
    assert(i < Matrix4f::ROWS);

    return (float *)&s[i];
}

const float *open3d::Matrix4f::operator[](const uint &i) const {
    assert(i < Matrix4f::ROWS);

    return (const float *const) & s[i];
}
