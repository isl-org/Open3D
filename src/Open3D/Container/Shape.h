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
#include <vector>

namespace open3d {

// Ref:
// https://github.com/NervanaSystems/ngraph/blob/master/src/ngraph/shape.hpp
class Shape : public std::vector<size_t> {
public:
    Shape(const std::initializer_list<size_t>& dim_sizes)
        : std::vector<size_t>(dim_sizes) {}

    Shape(const std::vector<size_t>& dim_sizes)
        : std::vector<size_t>(dim_sizes) {}

    Shape(const Shape& other) : std::vector<size_t>(other) {}

    explicit Shape(size_t n, size_t initial_value = 0)
        : std::vector<size_t>(n, initial_value) {}

    template <class InputIterator>
    Shape(InputIterator first, InputIterator last)
        : std::vector<size_t>(first, last) {}

    Shape() {}

    Shape& operator=(const Shape& v) {
        static_cast<std::vector<size_t>*>(this)->operator=(v);
        return *this;
    }

    Shape& operator=(Shape&& v) {
        static_cast<std::vector<size_t>*>(this)->operator=(v);
        return *this;
    }

    size_t NumElements() const {
        if (this->size() == 0) {
            return 0;
        }
        size_t size = 1;
        for (const size_t& d : *this) {
            size *= d;
        }
        return size;
    }
};

}  // namespace open3d
