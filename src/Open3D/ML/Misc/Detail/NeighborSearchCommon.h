// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "nanoflann.hpp"

namespace open3d {
namespace ml {
namespace detail {

/// Supported metrics
enum Metric { L1, L2, Linf };

/// Adaptor for nanoflann
template <class T>
class Adaptor {
public:
    Adaptor(size_t num_points, const T* const data)
        : num_points(num_points), data(data) {}

    inline size_t kdtree_get_point_count() const { return num_points; }

    inline T kdtree_get_pt(const size_t idx, int dim) const {
        return data[3 * idx + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

private:
    size_t num_points;
    const T* const data;
};

template <int METRIC, class T>
struct SelectNanoflannAdaptor {};

template <class T>
struct SelectNanoflannAdaptor<L2, T> {
    typedef nanoflann::L2_Adaptor<T, Adaptor<T>> Adaptor_t;
};

template <class T>
struct SelectNanoflannAdaptor<L1, T> {
    typedef nanoflann::L1_Adaptor<T, Adaptor<T>> Adaptor_t;
};

}  // namespace detail
}  // namespace ml
}  // namespace open3d
