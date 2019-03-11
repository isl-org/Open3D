
#include "Matrix6d.h"

#include <cassert>

double *open3d::Matrix6d::operator[](const uint &i) {
    assert(i < Matrix6d::ROWS);

    return (double *)&s[i];
}

const double *open3d::Matrix6d::operator[](const uint &i) const {
    assert(i < Matrix6d::ROWS);

    return (const double *const) & s[i];
}

open3d::Matrix6d::operator double *const() {
    return reinterpret_cast<double *>(s);
}

open3d::Matrix6d::operator const double *const() {
    return reinterpret_cast<const double *const>(s);
}
