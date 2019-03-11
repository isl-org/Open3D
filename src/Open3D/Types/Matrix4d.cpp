
#include "Matrix4d.h"

#include <cassert>

double *open3d::Matrix4d::operator[](const uint &i) {
    assert(i < Matrix4d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix4d::operator[](const uint &i) const {
    assert(i < Matrix4d::ROWS);

    return (const double *const) & s[i];
}
