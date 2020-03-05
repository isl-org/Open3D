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

/// \brief Wrap around negative \p dim.
///
/// E.g. If max_dim == 5, dim -1 will be converted to 4.
///
/// \param dim Dimension index
/// \param max_dim Maximum dimension index
/// \param inclusive Set to true to allow dim == max_dim. E.g. for slice
///        T[start:end], we allow end == max_dim.
static inline int64_t WrapDim(int64_t dim,
                              int64_t max_dim,
                              bool inclusive = false) {
    if (max_dim <= 0) {
        utility::LogError("max_dim {} must be >= 0");
    }
    int64_t min = -max_dim;
    int64_t max = inclusive ? max_dim : max_dim - 1;

    if (dim < min || dim > max) {
        utility::LogError(
                "Index out-of-range: dim == {}, but it must satisfy {} <= dim "
                "<= {}",
                dim, min, max);
    }
    if (dim < 0) {
        dim += max_dim;
    }
    return dim;
}

// Infers the size of a dim with size -1, if it exists. Also checks that new
// shape is compatible with the number of elements.
//
// E.g. Shape({2, -1, 4}) with num_elemnts 24, will be inferred as {2, 3, 4}.
//
// Ref: PyTorch's aten/src/ATen/InferSize.h
inline SizeVector InferShape(SizeVector shape, int64_t num_elements) {
    SizeVector inferred_shape = shape;
    int64_t new_size = 1;
    bool has_inferred_dim = false;
    int64_t inferred_dim;
    for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
        if (shape[dim] == -1) {
            if (has_inferred_dim) {
                utility::LogError(
                        "Proposed shape {}, but at most one dimension can be "
                        "-1 (inferred).",
                        shape.ToString());
            }
            inferred_dim = dim;
        } else if (shape[dim] >= 0) {
            new_size *= shape[dim];
        } else {
            utility::LogError("Invalid shape dimension {}", shape[dim]);
        }
    }

    if (num_elements == new_size ||
        (has_inferred_dim && new_size > 0 && num_elements % new_size == 0)) {
        if (has_inferred_dim) {
            // We have a degree of freedom here to select the dimension size;
            // follow NumPy semantics and just bail.  However, a nice error
            // message is needed because users often use `view` as a way to
            // flatten & unflatten dimensions and will otherwise be confused why
            //   empty_tensor.view( 0, 0)
            // works yet
            //   empty_tensor.view(-1, 0)
            // doesn't.
            if (new_size == 0) {
                utility::LogError(
                        "Cannot reshape tensor of 0 elements into shape {}, "
                        "because the unspecified dimension size -1 can be any "
                        "value and is ambiguous.",
                        shape.ToString());
            }
            inferred_shape[inferred_dim] = num_elements / new_size;
        }
        return inferred_shape;
    }

    utility::LogError("Shape {} is invalid for {} number of elements.", shape,
                      num_elements);
}

}  // namespace open3d
