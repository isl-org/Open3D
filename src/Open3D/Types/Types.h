#pragma once

#include "Matrix3d.h"
#include "Matrix3f.h"

#include "Matrix4d.h"
#include "Matrix4f.h"

#include "Matrix6d.h"
#include "Matrix6f.h"

#include "Vector3d.h"
#include "Vector3f.h"
#include "Vector3i.h"

namespace open3d {
template <typename T>
bool operator==(const T& m0, const T& m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            if (m0[r][c] != m1[r][c]) return false;

    return true;
}
template <typename T>
bool operator!=(const T& m0, const T& m1) {
    return !(m0 == m1);
}
template <typename T>
bool operator<=(const T& m0, const T& m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            if (m0[r][c] > m1[r][c]) return false;

    return true;
}
template <typename T>
bool operator>=(const T& m0, const T& m1) {
    static_assert(std::is_base_of<Matrix3d, T>::value ||
                          std::is_base_of<Matrix3f, T>::value ||
                          std::is_base_of<Matrix4d, T>::value ||
                          std::is_base_of<Matrix4f, T>::value ||
                          std::is_base_of<Matrix6d, T>::value ||
                          std::is_base_of<Matrix6f, T>::value,
                  "unsuported type");

    for (uint r = 0; r < Matrix3d::ROWS; r++)
        for (uint c = 0; c < Matrix3d::COLS; c++)
            if (m0[r][c] < m1[r][c]) return false;

    return true;
}
}  // namespace open3d
