
#include "Vector6d.h"

#include <cassert>

double &open3d::Vector6d::operator[](const uint &i) {
    assert(i < Vector6d::COLS);

    return s[0][i];
}

const double &open3d::Vector6d::operator[](const uint &i) const {
    assert(i < Vector6d::COLS);

    return s[0][i];
}
