// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#include <sstream>
#include <string>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {

/// DynamicSizeVector is a vector of optional<int64_t>, it is used to represent
/// a shape with unknown (dynamic) dimensions.
///
/// Example: create a shape of (None, 3)
/// ```
/// core::DynamicSizeVector shape{utility::nullopt, 3};
/// ```
class DynamicSizeVector : public std::vector<utility::optional<int64_t>> {
    using optint64_t = utility::optional<int64_t>;

public:
    DynamicSizeVector(const std::initializer_list<optint64_t>& dim_sizes)
        : std::vector<optint64_t>(dim_sizes) {}

    DynamicSizeVector(const std::vector<optint64_t>& dim_sizes)
        : std::vector<optint64_t>(dim_sizes) {}

    DynamicSizeVector(const DynamicSizeVector& other)
        : std::vector<optint64_t>(other) {}

    explicit DynamicSizeVector(int64_t n, int64_t initial_value = 0)
        : std::vector<optint64_t>(n, initial_value) {}

    template <class InputIterator>
    DynamicSizeVector(InputIterator first, InputIterator last)
        : std::vector<optint64_t>(first, last) {}

    DynamicSizeVector() {}

    DynamicSizeVector& operator=(const DynamicSizeVector& v) {
        static_cast<std::vector<optint64_t>*>(this)->operator=(v);
        return *this;
    }

    DynamicSizeVector& operator=(DynamicSizeVector&& v) {
        static_cast<std::vector<optint64_t>*>(this)->operator=(v);
        return *this;
    }

    std::string ToString() const {
        std::stringstream ss;
        ss << "{";
        bool first = true;
        for (const optint64_t& element : *this) {
            if (first) {
                first = false;
            } else {
                ss << ", ";
            }
            if (element.has_value()) {
                ss << fmt::format("{}", element.value());
            } else {
                ss << "None";
            }
        }
        ss << "}";
        return ss.str();
    }
};

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
        return std::accumulate(
                this->begin(), this->end(), 1LL,
                [this](const int64_t& lhs, const int64_t& rhs) -> int64_t {
                    if (lhs < 0 || rhs < 0) {
                        utility::LogError(
                                "Shape {} cannot contain negative dimensions.",
                                this->ToString());
                    }
                    return std::multiplies<int64_t>()(lhs, rhs);
                });
    }

    int64_t GetLength() const {
        if (size() == 0) {
            utility::LogError("Cannot get length of a 0-dimensional shape.");
        } else {
            return operator[](0);
        }
    }

    std::string ToString() const { return fmt::format("{}", *this); }

    void AssertCompatible(const DynamicSizeVector& dsv,
                          const std::string msg = "") const {
        if (!IsCompatible(dsv)) {
            if (msg.empty()) {
                utility::LogError("Shape {} is not compatible with {}.",
                                  ToString(), dsv.ToString());
            } else {
                utility::LogError("Shape {} is not compatible with {}: {}",
                                  ToString(), dsv.ToString(), msg);
            }
        }
    }

    bool IsCompatible(const DynamicSizeVector& dsv) const {
        if (size() != dsv.size()) {
            return false;
        }
        for (size_t i = 0; i < size(); ++i) {
            if (dsv[i].has_value() && dsv[i].value() != at(i)) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace core
}  // namespace open3d
