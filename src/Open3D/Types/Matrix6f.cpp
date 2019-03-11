
#include "Matrix6f.h"

#include <cassert>

float *open3d::Matrix6f::operator[](const uint &i) {
    assert(i < Matrix6f::ROWS);

    return (float *)&s[i];
}

const float *open3d::Matrix6f::operator[](const uint &i) const {
    assert(i < Matrix6f::ROWS);

    return (const float *const) & s[i];
}

open3d::Matrix6f::operator float *const() {
    return reinterpret_cast<float *>(s);
}

open3d::Matrix6f::operator const float *const() {
    return reinterpret_cast<const float *const>(s);
}
