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

#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "Open3D/Utility/Console.h"

namespace open3d {

/// SizeVector is a vector of int64_t, typically used in Tensor shape and
/// strides. A signed int64_t type is chosen to allow negative strides.
class SizeVector : public std::vector<int64_t> {
public:
    SizeVector(const std::initializer_list<int64_t>& dim_sizes)
        : std::vector<int64_t>(dim_sizes) {}

    SizeVector(const std::vector<int64_t>& dim_sizes)
        : std::vector<int64_t>(dim_sizes) {}

    SizeVector(const SizeVector& other) : std::vector<int64_t>(other) {}

    explicit SizeVector(int64_t n, int64_t initial_value = 0)
        : std::vector<int64_t>(n, initial_value) {}

    template <class InputIterator>
    SizeVector(InputIterator first, InputIterator last)
        : std::vector<int64_t>(first, last) {}

    SizeVector() {}

    SizeVector& operator=(const SizeVector& v) {
        static_cast<std::vector<int64_t>*>(this)->operator=(v);
        return *this;
    }

    SizeVector& operator=(SizeVector&& v) {
        static_cast<std::vector<int64_t>*>(this)->operator=(v);
        return *this;
    }

    int64_t NumElements() const {
        if (this->size() == 0) {
            return 1;
        }
        return std::accumulate(this->begin(), this->end(), 1,
                               std::multiplies<int64_t>());
    }

    std::string ToString() const { return fmt::format("{}", *this); }
};

}  // namespace open3d
