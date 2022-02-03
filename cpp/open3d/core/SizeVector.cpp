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

#include "open3d/core/SizeVector.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {

DynamicSizeVector::DynamicSizeVector(
        const std::initializer_list<utility::optional<int64_t>>& dim_sizes)
    : std::vector<utility::optional<int64_t>>(dim_sizes) {}

DynamicSizeVector::DynamicSizeVector(
        const std::vector<utility::optional<int64_t>>& dim_sizes)
    : std::vector<utility::optional<int64_t>>(dim_sizes) {}

DynamicSizeVector::DynamicSizeVector(const DynamicSizeVector& other)
    : std::vector<utility::optional<int64_t>>(other) {}

DynamicSizeVector::DynamicSizeVector(int64_t n, int64_t initial_value)
    : std::vector<utility::optional<int64_t>>(n, initial_value) {}

DynamicSizeVector::DynamicSizeVector(const SizeVector& dim_sizes)
    : DynamicSizeVector(dim_sizes.begin(), dim_sizes.end()) {}

SizeVector DynamicSizeVector::ToSizeVector() const {
    SizeVector sv(size());
    std::transform(begin(), end(), sv.begin(), [](const auto& v) {
        if (!v.has_value()) {
            utility::LogError("Cannot convert dynamic shape to SizeVector.");
        }
        return v.value();
    });
    return sv;
}

DynamicSizeVector& DynamicSizeVector::operator=(const DynamicSizeVector& v) {
    static_cast<std::vector<utility::optional<int64_t>>*>(this)->operator=(v);
    return *this;
}

DynamicSizeVector& DynamicSizeVector::operator=(DynamicSizeVector&& v) {
    static_cast<std::vector<utility::optional<int64_t>>*>(this)->operator=(v);
    return *this;
}

std::string DynamicSizeVector::ToString() const {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const utility::optional<int64_t>& element : *this) {
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

bool DynamicSizeVector::IsDynamic() const {
    return std::any_of(
            this->begin(), this->end(),
            [](const utility::optional<int64_t>& v) { return !v.has_value(); });
}

SizeVector::SizeVector(const std::initializer_list<int64_t>& dim_sizes)
    : std::vector<int64_t>(dim_sizes) {}

SizeVector::SizeVector(const std::vector<int64_t>& dim_sizes)
    : std::vector<int64_t>(dim_sizes) {}

SizeVector::SizeVector(const SizeVector& other) : std::vector<int64_t>(other) {}

SizeVector::SizeVector(int64_t n, int64_t initial_value)
    : std::vector<int64_t>(n, initial_value) {}

SizeVector& SizeVector::operator=(const SizeVector& v) {
    static_cast<std::vector<int64_t>*>(this)->operator=(v);
    return *this;
}

SizeVector& SizeVector::operator=(SizeVector&& v) {
    static_cast<std::vector<int64_t>*>(this)->operator=(v);
    return *this;
}

int64_t SizeVector::NumElements() const {
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

int64_t SizeVector::GetLength() const {
    if (size() == 0) {
        utility::LogError("Cannot get length of a 0-dimensional shape.");
    } else {
        return operator[](0);
    }
}

std::string SizeVector::ToString() const { return fmt::format("{}", *this); }

void SizeVector::AssertCompatible(const DynamicSizeVector& dsv,
                                  const std::string msg) const {
    if (!IsCompatible(dsv)) {
        if (msg.empty()) {
            utility::LogError("Shape {} is not compatible with {}.", ToString(),
                              dsv.ToString());
        } else {
            utility::LogError("Shape {} is not compatible with {}: {}",
                              ToString(), dsv.ToString(), msg);
        }
    }
}

bool SizeVector::IsCompatible(const DynamicSizeVector& dsv) const {
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

}  // namespace core
}  // namespace open3d
