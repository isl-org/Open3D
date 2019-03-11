
#include "Vector4d.h"

#include <cassert>

double &open3d::Vector4d::operator[](const uint &i) {
    assert(i < Vector4d::COLS);

    return s[0][i];
}

const double &open3d::Vector4d::operator[](const uint &i) const {
    assert(i < Vector4d::COLS);

    return s[0][i];
}
