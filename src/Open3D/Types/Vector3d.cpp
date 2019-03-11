
#include "Vector3d.h"

#include <cassert>

double &open3d::Vector3d::operator[](const uint &i) {
    assert(i < Vector3d::COLS);

    return s[0][i];
}

const double &open3d::Vector3d::operator[](const uint &i) const {
    assert(i < Vector3d::COLS);

    return s[0][i];
}
