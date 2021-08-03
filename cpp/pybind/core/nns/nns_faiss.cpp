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

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/FaissIndex.h"
#include "pybind/core/nns/nearest_neighbor_search.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {
namespace nns {

void pybind_core_faiss(py::module &m_nns) {
    py::class_<FaissIndex, std::shared_ptr<FaissIndex>> faiss(
            m_nns, "FaissIndex",
            "FaissIndex class for nearest neighbor search. "
            "Construct a NearestNeighborSearch object with input "
            "dataset_points of shape {n_dataset, d}.");

    // Constructors.
    faiss.def(py::init<>());

    // Index functions.
    faiss.def(
            "set_tensor_data",
            [](FaissIndex &self, Tensor dataset_points) {
                return self.SetTensorData(dataset_points);
            },
            py::arg("dataset_points"));

    // Search functions.
    faiss.def(
            "knn_search",
            [](FaissIndex &self, Tensor query_points, int knn) {
                return self.SearchKnn(query_points, knn);
            },
            py::arg("query_points"), py::arg("knn"));
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
