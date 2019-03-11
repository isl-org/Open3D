
#include "Vector3i.h"

#include <cassert>

int &open3d::Vector3i::operator[](const uint &i) {
    assert(i < Vector3i::COLS);

    return s[0][i];
}

const int &open3d::Vector3i::operator[](const uint &i) const {
    assert(i < Vector3i::COLS);

    return s[0][i];
}
