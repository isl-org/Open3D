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

#include "open3d/ml/contrib/contrib_nns.h"

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/ml/contrib/contrib.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {
namespace contrib {

void pybind_contrib_nns(py::module& m_contrib) {
    m_contrib.def("knn_search", &KnnSearch, "query_points"_a,
                  "dataset_points"_a, "knn"_a);
    m_contrib.def("radius_search", &RadiusSearch, "query_points"_a,
                  "dataset_points"_a, "query_batches"_a, "dataset_batches"_a,
                  "radius"_a);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
