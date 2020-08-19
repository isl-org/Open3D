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

#include "open3d/geometry/KnnFaiss.h"

#include "pybind/docstring.h"
#include "pybind/geometry/knnfaiss/knnfaiss.h"
//#include "pybind/geometry/geometry.h"
//#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {

void pybind_knnfaiss(py::module &m) {
    // open3d.geometry.KnnFaiss
    static const std::unordered_map<std::string, std::string>
            map_knn_faiss_method_docs = {
                    {"query", "The input query point."},
                    {"radius", "Search radius."},
                    {"max_nn",
                     "At maximum, ``max_nn`` neighbors will be searched."},
                    {"knn", "``knn`` neighbors will be searched."},
                    {"feature", "Feature data."},
                    {"data", "Matrix data."},
                    {"tensor", "Tensor data."}};
    py::class_<geometry::KnnFaiss, std::shared_ptr<geometry::KnnFaiss>>
            knnfaiss(m, "KnnFaiss", "Faiss for nearest neighbor search.");
    knnfaiss.def(py::init<>())
            .def(py::init<const Eigen::MatrixXd &>(), "data"_a)
            .def("set_matrix_data", &geometry::KnnFaiss::SetMatrixData,
                 "Sets the data for the Faiss Index from a matrix.", "data"_a)
            .def(py::init<const core::Tensor &>(), "tensor"_a)
            .def("set_tensor_data", &geometry::KnnFaiss::SetTensorData,
                 "Sets the data for the Faiss Index from a tensor.", "tensor"_a)
            .def(py::init<const geometry::Geometry &>(), "geometry"_a)
            .def("set_geometry", &geometry::KnnFaiss::SetGeometry,
                 "Sets the data for the Faiss Index from geometry.",
                 "geometry"_a)
            .def(py::init<const pipelines::registration::Feature &>(),
                 "feature"_a)
            .def("set_feature", &geometry::KnnFaiss::SetFeature,
                 "Sets the data for the Faiss Index from the feature data.",
                 "feature"_a)
            .def(
                    "search_knn_vector_3d",
                    [](const geometry::KnnFaiss &index,
                       const Eigen::Vector3d &query, int knn) {
                        std::vector<int64_t> indices;
                        std::vector<float> distance2;
                        int k = index.SearchKNN(query, knn, indices, distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_knn_vector_3d() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "knn"_a)
            .def(
                    "search_radius_vector_3d",
                    [](const geometry::KnnFaiss &index,
                       const Eigen::Vector3d &query, float radius) {
                        std::vector<int64_t> indices;
                        std::vector<float> distance2;
                        int k = index.SearchRadius(query, radius, indices,
                                                   distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_radius_vector_3d() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "radius"_a)
            .def(
                    "search_knn_vector_xd",
                    [](const geometry::KnnFaiss &index,
                       const Eigen::VectorXd &query, int knn) {
                        std::vector<int64_t> indices;
                        std::vector<float> distance2;
                        int k = index.SearchKNN(query, knn, indices, distance2);

                        if (k < 0)
                            throw std::runtime_error(
                                    "search_knn_vector_xd() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "knn"_a)
            .def(
                    "search_radius_vector_xd",
                    [](const geometry::KnnFaiss &index,
                       const Eigen::VectorXd &query, float radius) {
                        std::vector<int64_t> indices;
                        std::vector<float> distance2;
                        int k = index.SearchRadius(query, radius, indices,
                                                   distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_radius_vector_xd() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "radius"_a);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "search_knn_vector_3d",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "search_knn_vector_xd",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "search_radius_vector_3d",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "search_radius_vector_xd",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "set_feature",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "set_geometry",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "set_matrix_data",
                                    map_knn_faiss_method_docs);
    docstring::ClassMethodDocInject(m, "KnnFaiss", "set_tensor_data",
                                    map_knn_faiss_method_docs);
}

}  // namespace open3d
