
#include "Matrix3d.h"

#include <cassert>

double *open3d::Matrix3d::operator[](const uint &i) {
    assert(i < Matrix3d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix3d::operator[](const uint &i) const {
    assert(i < Matrix3d::ROWS);

    return (const double *const) & s[i];
}
