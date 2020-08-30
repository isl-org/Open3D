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

#include "open3d/core/nns/NearestNeighborSearch.h"

#include "open3d/core/Tensor.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {

void pybind_core_nn(py::module &m) {
    py::module m_nn = m.def_submodule("nns");

    // open3d.core.nns.NearestNeighborSearch
    static const std::unordered_map<std::string, std::string>
            map_nearest_neighbor_search_method_docs = {
                    {"query", "The input query tensor."},
                    {"radii", "Search multiple radii"},
                    {"radius", "Search fixed radius."},
                    {"max_nn",
                     "At maximum, ``max_nn`` neighbors will be searched."},
                    {"knn", "``knn`` neighbors will be searched."}};
    py::class_<nns::NearestNeighborSearch,
               std::shared_ptr<nns::NearestNeighborSearch>>
            nearestneighbor(
                    m_nn, "NearestNeighborSearch",
                    "NearestNeighborSearch class for nearest neighbor search.");

    nearestneighbor.def(py::init<const Tensor &>(), "data"_a)
            .def("knn_index", &nns::NearestNeighborSearch::KnnIndex)
            .def("fixed_radius_index",
                 &nns::NearestNeighborSearch::FixedRadiusIndex)
            .def("multi_radius_index",
                 &nns::NearestNeighborSearch::MultiRadiusIndex)
            .def("hybrid_index", &nns::NearestNeighborSearch::HybridIndex)
            .def("knn_search", &nns::NearestNeighborSearch::KnnSearch,
                 "query"_a, "knn"_a)
            .def(
                    "multi_radius_search",
                    [](nns::NearestNeighborSearch &nn_, const Tensor &query,
                       py::array radii) {
                        Tensor radii_t = PyArrayToTensor(radii, true);
                        if (radii_t.GetDtype() != Dtype::Float64) {
                            utility::LogError("Radius type must be Float64!");
                        }
                        return nn_.MultiRadiusSearch(
                                query, radii_t.ToFlatVector<double>());
                    },
                    "query"_a, "radii"_a)
            .def("fixed_radius_search",
                 &nns::NearestNeighborSearch::FixedRadiusSearch, "query"_a,
                 "radius"_a)
            .def("hybrid_search", &nns::NearestNeighborSearch::HybridSearch,
                 "query"_a, "radius"_a, "max_knn"_a);
    docstring::ClassMethodDocInject(m_nn, "NearestNeighborSearch", "knn_search",
                                    map_nearest_neighbor_search_method_docs);
    docstring::ClassMethodDocInject(m_nn, "NearestNeighborSearch",
                                    "fixed_radius_search",
                                    map_nearest_neighbor_search_method_docs);
    docstring::ClassMethodDocInject(m_nn, "NearestNeighborSearch",
                                    "multi_radius_search",
                                    map_nearest_neighbor_search_method_docs);
    docstring::ClassMethodDocInject(m_nn, "NearestNeighborSearch",
                                    "hybrid_search",
                                    map_nearest_neighbor_search_method_docs);
}

}  // namespace core
}  // namespace open3d
