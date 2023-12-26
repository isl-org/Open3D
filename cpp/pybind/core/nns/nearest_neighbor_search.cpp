// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/core/nns/nearest_neighbor_search.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {
namespace nns {

void pybind_core_nns(py::module &m_nns) {
    static const std::unordered_map<std::string, std::string>
            map_nearest_neighbor_search_method_docs = {
                    {"query_points", "The query tensor of shape {n_query, d}."},
                    {"radii",
                     "Tensor of shape {n_query,} containing multiple radii, "
                     "one for each query point."},
                    {"radius", "Radius value for radius search."},
                    {"max_knn",
                     "Maximum number of neighbors to search per query point."},
                    {"knn", "Number of neighbors to search per query point."}};

    py::class_<NearestNeighborSearch, std::shared_ptr<NearestNeighborSearch>>
            nns(m_nns, "NearestNeighborSearch",
                "NearestNeighborSearch class for nearest neighbor search. "
                "Construct a NearestNeighborSearch object with input "
                "dataset_points of shape {n_dataset, d}.");

    // Constructors.
    nns.def(py::init<const Tensor &, const Dtype>(), "dataset_points"_a,
            "index_dtype"_a = core::Int64);

    // Index functions.
    nns.def("knn_index", &NearestNeighborSearch::KnnIndex,
            "Set index for knn search.");
    nns.def(
            "fixed_radius_index",
            [](NearestNeighborSearch &self, utility::optional<double> radius) {
                if (!radius.has_value()) {
                    return self.FixedRadiusIndex();
                } else {
                    return self.FixedRadiusIndex(radius.value());
                }
            },
            py::arg("radius") = py::none());
    nns.def("multi_radius_index", &NearestNeighborSearch::MultiRadiusIndex,
            "Set index for multi-radius search.");
    nns.def(
            "hybrid_index",
            [](NearestNeighborSearch &self, utility::optional<double> radius) {
                if (!radius.has_value()) {
                    return self.HybridIndex();
                } else {
                    return self.HybridIndex(radius.value());
                }
            },
            py::arg("radius") = py::none());

    // Search functions.
    nns.def("knn_search", &NearestNeighborSearch::KnnSearch, "query_points"_a,
            "knn"_a, "Perform knn search.");
    nns.def(
            "fixed_radius_search",
            [](NearestNeighborSearch &self, Tensor query_points, double radius,
               utility::optional<bool> sort) {
                if (!sort.has_value()) {
                    return self.FixedRadiusSearch(query_points, radius, true);
                } else {
                    return self.FixedRadiusSearch(query_points, radius,
                                                  sort.value());
                }
            },
            py::arg("query_points"), py::arg("radius"),
            py::arg("sort") = py::none());
    nns.def("multi_radius_search", &NearestNeighborSearch::MultiRadiusSearch,
            "query_points"_a, "radii"_a,
            "Perform multi-radius search. Each query point has an independent "
            "radius.");
    nns.def("hybrid_search", &NearestNeighborSearch::HybridSearch,
            "query_points"_a, "radius"_a, "max_knn"_a,
            "Perform hybrid search.");

    // Docstrings.
    docstring::ClassMethodDocInject(m_nns, "NearestNeighborSearch",
                                    "knn_search",
                                    map_nearest_neighbor_search_method_docs);
    docstring::ClassMethodDocInject(m_nns, "NearestNeighborSearch",
                                    "multi_radius_search",
                                    map_nearest_neighbor_search_method_docs);
    docstring::ClassMethodDocInject(m_nns, "NearestNeighborSearch",
                                    "fixed_radius_search",
                                    map_nearest_neighbor_search_method_docs);
    docstring::ClassMethodDocInject(m_nns, "NearestNeighborSearch",
                                    "hybrid_search",
                                    map_nearest_neighbor_search_method_docs);
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
