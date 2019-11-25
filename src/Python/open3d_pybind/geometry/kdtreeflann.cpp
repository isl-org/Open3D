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

#include "Open3D/Geometry/KDTreeFlann.h"

#include "open3d_pybind/docstring.h"
#include "open3d_pybind/geometry/geometry.h"
#include "open3d_pybind/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_kdtreeflann(py::module &m) {
    // open3d.geometry.KDTreeSearchParam
    py::class_<geometry::KDTreeSearchParam> kdtreesearchparam(
            m, "KDTreeSearchParam", "Base class for KDTree search parameters.");
    kdtreesearchparam.def("get_search_type",
                          &geometry::KDTreeSearchParam::GetSearchType,
                          "Get the search type (KNN, Radius, Hybrid) for the "
                          "search parameter.");
    docstring::ClassMethodDocInject(m, "KDTreeSearchParam", "get_search_type");

    // open3d.geometry.KDTreeSearchParam.Type
    py::enum_<geometry::KDTreeSearchParam::SearchType> kdtree_search_param_type(
            kdtreesearchparam, "Type", py::arithmetic());
    kdtree_search_param_type
            .value("KNNSearch", geometry::KDTreeSearchParam::SearchType::Knn)
            .value("RadiusSearch",
                   geometry::KDTreeSearchParam::SearchType::Radius)
            .value("HybridSearch",
                   geometry::KDTreeSearchParam::SearchType::Hybrid)
            .export_values();
    kdtree_search_param_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Geometry types.";
            }),
            py::none(), py::none(), "");

    // open3d.geometry.KDTreeSearchParamKNN
    py::class_<geometry::KDTreeSearchParamKNN> kdtreesearchparam_knn(
            m, "KDTreeSearchParamKNN", kdtreesearchparam,
            "KDTree search parameters for pure KNN search.");
    kdtreesearchparam_knn.def(py::init<int>(), "knn"_a = 30)
            .def("__repr__",
                 [](const geometry::KDTreeSearchParamKNN &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamKNN with knn "
                                    "= ") +
                            std::to_string(param.knn_);
                 })
            .def_readwrite("knn", &geometry::KDTreeSearchParamKNN::knn_,
                           "Number of the neighbors that will be searched.");

    // open3d.geometry.KDTreeSearchParamRadius
    py::class_<geometry::KDTreeSearchParamRadius> kdtreesearchparam_radius(
            m, "KDTreeSearchParamRadius", kdtreesearchparam,
            "KDTree search parameters for pure radius search.");
    kdtreesearchparam_radius.def(py::init<double>(), "radius"_a)
            .def("__repr__",
                 [](const geometry::KDTreeSearchParamRadius &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamRadius with "
                                    "radius = ") +
                            std::to_string(param.radius_);
                 })
            .def_readwrite("radius",
                           &geometry::KDTreeSearchParamRadius::radius_,
                           "Search radius.");

    // open3d.geometry.KDTreeSearchParamHybrid
    py::class_<geometry::KDTreeSearchParamHybrid> kdtreesearchparam_hybrid(
            m, "KDTreeSearchParamHybrid", kdtreesearchparam,
            "KDTree search parameters for hybrid KNN and radius search.");
    kdtreesearchparam_hybrid
            .def(py::init<double, int>(), "radius"_a, "max_nn"_a)
            .def("__repr__",
                 [](const geometry::KDTreeSearchParamHybrid &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamHybrid with "
                                    "radius = ") +
                            std::to_string(param.radius_) +
                            " and max_nn = " + std::to_string(param.max_nn_);
                 })
            .def_readwrite("radius",
                           &geometry::KDTreeSearchParamHybrid::radius_,
                           "Search radius.")
            .def_readwrite(
                    "max_nn", &geometry::KDTreeSearchParamHybrid::max_nn_,
                    "At maximum, ``max_nn`` neighbors will be searched.");

    // open3d.geometry.KDTreeFlann
    static const std::unordered_map<std::string, std::string>
            map_kd_tree_flann_method_docs = {
                    {"query", "The input query point."},
                    {"radius", "Search radius."},
                    {"max_nn",
                     "At maximum, ``max_nn`` neighbors will be searched."},
                    {"knn", "``knn`` neighbors will be searched."},
                    {"feature", "Feature data."},
                    {"data", "Matrix data."}};
    py::class_<geometry::KDTreeFlann, std::shared_ptr<geometry::KDTreeFlann>>
            kdtreeflann(m, "KDTreeFlann",
                        "KDTree with FLANN for nearest neighbor search.");
    kdtreeflann.def(py::init<>())
            .def(py::init<const Eigen::MatrixXd &>(), "data"_a)
            .def("set_matrix_data", &geometry::KDTreeFlann::SetMatrixData,
                 "Sets the data for the KDTree from a matrix.", "data"_a)
            .def(py::init<const geometry::Geometry &>(), "geometry"_a)
            .def("set_geometry", &geometry::KDTreeFlann::SetGeometry,
                 "Sets the data for the KDTree from geometry.", "geometry"_a)
            .def(py::init<const registration::Feature &>(), "feature"_a)
            .def("set_feature", &geometry::KDTreeFlann::SetFeature,
                 "Sets the data for the KDTree from the feature data.",
                 "feature"_a)
            // Although these C++ style functions are fast by orders of
            // magnitudes when similar queries are performed for a large number
            // of times and memory management is involved, we prefer not to
            // expose them in Python binding. Considering writing C++ functions
            // if performance is an issue.
            //.def("search_vector_3d_in_place",
            //&KDTreeFlann::Search<Eigen::Vector3d>,
            //        "query"_a, "search_param"_a, "indices"_a, "distance2"_a)
            //.def("search_knn_vector_3d_in_place",
            //        &KDTreeFlann::SearchKNN<Eigen::Vector3d>,
            //        "query"_a, "knn"_a, "indices"_a, "distance2"_a)
            //.def("search_radius_vector_3d_in_place",
            //        &KDTreeFlann::SearchRadius<Eigen::Vector3d>, "query"_a,
            //        "radius"_a, "indices"_a, "distance2"_a)
            //.def("search_hybrid_vector_3d_in_place",
            //        &KDTreeFlann::SearchHybrid<Eigen::Vector3d>, "query"_a,
            //        "radius"_a, "max_nn"_a, "indices"_a, "distance2"_a)
            .def("search_vector_3d",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::Vector3d &query,
                    const geometry::KDTreeSearchParam &param) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.Search(query, param, indices, distance2);
                     if (k < 0)
                         throw std::runtime_error("search_vector_3d() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "search_param"_a)
            .def("search_knn_vector_3d",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::Vector3d &query, int knn) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.SearchKNN(query, knn, indices, distance2);
                     if (k < 0)
                         throw std::runtime_error(
                                 "search_knn_vector_3d() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "knn"_a)
            .def("search_radius_vector_3d",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::Vector3d &query, double radius) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.SearchRadius(query, radius, indices,
                                               distance2);
                     if (k < 0)
                         throw std::runtime_error(
                                 "search_radius_vector_3d() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "radius"_a)
            .def("search_hybrid_vector_3d",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::Vector3d &query, double radius, int max_nn) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.SearchHybrid(query, radius, max_nn, indices,
                                               distance2);
                     if (k < 0)
                         throw std::runtime_error(
                                 "search_hybrid_vector_3d() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "radius"_a, "max_nn"_a)
            .def("search_vector_xd",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::VectorXd &query,
                    const geometry::KDTreeSearchParam &param) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.Search(query, param, indices, distance2);
                     if (k < 0)
                         throw std::runtime_error("search_vector_xd() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "search_param"_a)
            .def("search_knn_vector_xd",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::VectorXd &query, int knn) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.SearchKNN(query, knn, indices, distance2);
                     if (k < 0)
                         throw std::runtime_error(
                                 "search_knn_vector_xd() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "knn"_a)
            .def("search_radius_vector_xd",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::VectorXd &query, double radius) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.SearchRadius(query, radius, indices,
                                               distance2);
                     if (k < 0)
                         throw std::runtime_error(
                                 "search_radius_vector_xd() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "radius"_a)
            .def("search_hybrid_vector_xd",
                 [](const geometry::KDTreeFlann &tree,
                    const Eigen::VectorXd &query, double radius, int max_nn) {
                     std::vector<int> indices;
                     std::vector<double> distance2;
                     int k = tree.SearchHybrid(query, radius, max_nn, indices,
                                               distance2);
                     if (k < 0)
                         throw std::runtime_error(
                                 "search_hybrid_vector_xd() error!");
                     return std::make_tuple(k, indices, distance2);
                 },
                 "query"_a, "radius"_a, "max_nn"_a);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_hybrid_vector_3d",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_hybrid_vector_xd",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_knn_vector_3d",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_knn_vector_xd",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_radius_vector_3d",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_radius_vector_xd",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_vector_3d",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_vector_xd",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "set_feature",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "set_geometry",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "set_matrix_data",
                                    map_kd_tree_flann_method_docs);
}
