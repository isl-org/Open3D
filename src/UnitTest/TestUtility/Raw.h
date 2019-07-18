// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace unit_test {
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
}  // namespace unit_test
