// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace open3d {
namespace tests {

// Class for "generating" data.
class Raw {
public:
    Raw() : step(1), index(0) {}
    Raw(const int &seed)
        : step((seed <= 0) ? 1 : seed), index(abs(seed) % SIZE) {}

private:
    // size of the raw data
    static const int SIZE = 1021;

    // raw data
    static std::vector<uint8_t> data_;

public:
    // low end of the range
    static const uint8_t VMIN = 0;

    // high end of the range
    static const uint8_t VMAX = 255;

private:
    // step through the raw data
    int step;

    // index into the raw data
    int index;

public:
    // Get the next value.
    template <class T>
    T Next();
};

// Get the next uint8_t value.
// Output range: [0, 255].
template <>
uint8_t Raw::Next();

// Get the next int value.
// Output range: [0, 255].
template <>
int Raw::Next();

// Get the next size_t value.
// Output range: [0, 255].
template <>
size_t Raw::Next();

// Get the next float value.
// Output range: [0, 1].
template <>
float Raw::Next();

// Get the next double value.
// Output range: [0, 1].
template <>
double Raw::Next();

}  // namespace tests
}  // namespace open3d
