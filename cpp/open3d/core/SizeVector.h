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
#include <string>
#include <vector>

#include "open3d/core/SmallVector.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {

class SizeVector;

/// DynamicSizeVector is a SmallVector of optional<int64_t>, it is used to
/// represent a shape with unknown (dynamic) dimensions. Dimensions up to size 4
/// are stored on the stack, and larger vectors are stored on the heap
/// automatically.
///
/// Example: create a shape of (None, 3)
/// ```
/// core::DynamicSizeVector shape{utility::nullopt, 3};
/// ```
class DynamicSizeVector : public SmallVector<utility::optional<int64_t>, 4> {
public:
    using super_t = SmallVector<utility::optional<int64_t>, 4>;
    DynamicSizeVector() {}

    DynamicSizeVector(
            const std::initializer_list<utility::optional<int64_t>>& dim_sizes);

    DynamicSizeVector(const std::vector<utility::optional<int64_t>>& dim_sizes);

    DynamicSizeVector(const DynamicSizeVector& other);

    explicit DynamicSizeVector(int64_t n, int64_t initial_value = 0);

    template <class InputIterator>
    DynamicSizeVector(InputIterator first, InputIterator last)
        : super_t(first, last) {}

    DynamicSizeVector(const SizeVector& dim_sizes);

    SizeVector ToSizeVector() const;

    DynamicSizeVector& operator=(const DynamicSizeVector& v);

    DynamicSizeVector& operator=(DynamicSizeVector&& v);

    std::string ToString() const;

    bool IsDynamic() const;

    // required for pybind
    void shrink_to_fit() {}
};

/// SizeVector is a SmallVector of int64_t, typically used in Tensor shape and
/// strides. Dimensions up to size 4 are stored on the stack, and larger vectors
/// are stored on the heap automatically.  A signed int64_t type is chosen to
/// allow negative strides.
class SizeVector : public SmallVector<int64_t, 4> {
public:
    using super_t = SmallVector<int64_t, 4>;
    SizeVector() {}

    SizeVector(const std::initializer_list<int64_t>& dim_sizes);

    SizeVector(const std::vector<int64_t>& dim_sizes);

    SizeVector(const SizeVector& other);

    explicit SizeVector(int64_t n, int64_t initial_value = 0);

    template <class InputIterator>
    SizeVector(InputIterator first, InputIterator last)
        : super_t(first, last) {}

    SizeVector& operator=(const SizeVector& v);

    SizeVector& operator=(SizeVector&& v);

    int64_t NumElements() const;

    int64_t GetLength() const;

    std::string ToString() const;

    void AssertCompatible(const DynamicSizeVector& dsv,
                          const std::string msg = "") const;

    bool IsCompatible(const DynamicSizeVector& dsv) const;

    operator std::vector<int64_t>() const {
        return std::vector<int64_t>(begin(), end());
    }

    // required for pybind
    void shrink_to_fit() {}
};

}  // namespace core
}  // namespace open3d
