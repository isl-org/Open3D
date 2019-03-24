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

#include <random>

namespace open3d {
namespace utility {

template <typename T>
struct Query {};

template <>
struct Query<double> {
    typedef std::uniform_real_distribution<double> Type;
};

template <>
struct Query<float> {
    typedef std::uniform_real_distribution<float> Type;
};

template <>
struct Query<int> {
    typedef std::uniform_int_distribution<int> Type;
};

template<typename T>
class Random {

};

template<typename T>
T Next(const T &min = (T)-1, const T &max = (T)1) {
    T output{};

    // setup randomness machine
    std::random_device device;
    std::mt19937 engine(device());
    typename Query<T>::Type query(min, max);

    return query(engine);
}
}  // namespace utility
}  // namespace open3d
