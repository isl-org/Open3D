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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Geometry/KDTreeFlann.h>
using namespace three;

void pybind_kdtreeflann(py::module &m)
{
	py::class_<KDTreeSearchParam> kdtreesearchparam(m, "KDTreeSearchParam");
	kdtreesearchparam
		.def("get_search_type", &KDTreeSearchParam::GetSearchType);
	py::enum_<KDTreeSearchParam::SearchType>(kdtreesearchparam, "Type",
			py::arithmetic())
		.value("KNNSearch", KDTreeSearchParam::SearchType::Knn)
		.value("RadiusSearch", KDTreeSearchParam::SearchType::Radius)
		.value("HybridSearch", KDTreeSearchParam::SearchType::Hybrid)
		.export_values();

	py::class_<KDTreeSearchParamKNN> kdtreesearchparam_knn(m,
			"KDTreeSearchParamKNN", kdtreesearchparam);
	kdtreesearchparam_knn
		.def(py::init<int>(), "knn"_a = 30)
		.def("__repr__", [](const KDTreeSearchParamKNN &param){
			return std::string("KDTreeSearchParamKNN with knn = ") +
					std::to_string(param.knn_);
		})
		.def_readwrite("knn", &KDTreeSearchParamKNN::knn_);

	py::class_<KDTreeSearchParamRadius> kdtreesearchparam_radius(m,
			"KDTreeSearchParamRadius", kdtreesearchparam);
	kdtreesearchparam_radius
		.def(py::init<double>(), "radius"_a)
		.def("__repr__", [](const KDTreeSearchParamRadius &param) {
			return std::string("KDTreeSearchParamRadius with radius = ") +
					std::to_string(param.radius_);
		})
		.def_readwrite("radius", &KDTreeSearchParamRadius::radius_);

	py::class_<KDTreeSearchParamHybrid> kdtreesearchparam_hybrid(m,
			"KDTreeSearchParamHybrid", kdtreesearchparam);
	kdtreesearchparam_hybrid
		.def(py::init<double, int>(), "radius"_a, "max_nn"_a)
		.def("__repr__", [](const KDTreeSearchParamHybrid &param) {
			return std::string("KDTreeSearchParamHybrid with radius = ") +
					std::to_string(param.radius_) + " and max_nn = " +
					std::to_string(param.max_nn_);
		})
		.def_readwrite("radius", &KDTreeSearchParamHybrid::radius_)
		.def_readwrite("max_nn", &KDTreeSearchParamHybrid::max_nn_);

	py::class_<KDTreeFlann, std::shared_ptr<KDTreeFlann>> kdtreeflann(m,
			"KDTreeFlann");
	kdtreeflann.def(py::init<>())
		.def(py::init<const Eigen::MatrixXd &>(), "data"_a)
		.def("set_matrix_data", &KDTreeFlann::SetMatrixData, "data"_a)
		.def(py::init<const Geometry &>(), "geometry"_a)
		.def("set_geometry", &KDTreeFlann::SetGeometry, "geometry"_a)
		.def(py::init<const Feature &>(), "feature"_a)
		.def("set_feature", &KDTreeFlann::SetFeature, "feature"_a)
		// Although these C++ style functions are fast by orders of magnitudes
		// when similar queries are performed for a large number of times and
		// memory management is involved, we prefer not to expose them in
		// Python binding. Considering writing C++ functions if performance is
		// an issue.
		//.def("search_vector_3d_in_place", &KDTreeFlann::Search<Eigen::Vector3d>,
		//		"query"_a, "search_param"_a, "indices"_a, "distance2"_a)
		//.def("search_knn_vector_3d_in_place",
		//		&KDTreeFlann::SearchKNN<Eigen::Vector3d>,
		//		"query"_a, "knn"_a, "indices"_a, "distance2"_a)
		//.def("search_radius_vector_3d_in_place",
		//		&KDTreeFlann::SearchRadius<Eigen::Vector3d>, "query"_a,
		//		"radius"_a, "indices"_a, "distance2"_a)
		//.def("search_hybrid_vector_3d_in_place",
		//		&KDTreeFlann::SearchHybrid<Eigen::Vector3d>, "query"_a,
		//		"radius"_a, "max_nn"_a, "indices"_a, "distance2"_a)
		.def("search_vector_3d", [](const KDTreeFlann &tree,
				const Eigen::Vector3d &query, const KDTreeSearchParam &param) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.Search(query, param, indices, distance2);
				if (k < 0)
					throw std::runtime_error("search_vector_3d() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "search_param"_a)
		.def("search_knn_vector_3d", [](const KDTreeFlann &tree,
				const Eigen::Vector3d &query, int knn) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.SearchKNN(query, knn, indices, distance2);
				if (k < 0)
					throw std::runtime_error("search_knn_vector_3d() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "knn"_a)
		.def("search_radius_vector_3d", [](const KDTreeFlann &tree,
				const Eigen::Vector3d &query, double radius) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.SearchRadius(query, radius, indices, distance2);
				if (k < 0)
					throw std::runtime_error("search_radius_vector_3d() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "radius"_a)
		.def("search_hybrid_vector_3d", [](const KDTreeFlann &tree,
				const Eigen::Vector3d &query, double radius, int max_nn) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.SearchHybrid(query, radius, max_nn, indices,
						distance2);
				if (k < 0)
					throw std::runtime_error("search_hybrid_vector_3d() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "radius"_a, "max_nn"_a)
		.def("search_vector_xd", [](const KDTreeFlann &tree,
				const Eigen::VectorXd &query, const KDTreeSearchParam &param) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.Search(query, param, indices, distance2);
				if (k < 0)
					throw std::runtime_error("search_vector_xd() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "search_param"_a)
		.def("search_knn_vector_xd", [](const KDTreeFlann &tree,
				const Eigen::VectorXd &query, int knn) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.SearchKNN(query, knn, indices, distance2);
				if (k < 0)
					throw std::runtime_error("search_knn_vector_xd() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "knn"_a)
		.def("search_radius_vector_xd", [](const KDTreeFlann &tree,
				const Eigen::VectorXd &query, double radius) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.SearchRadius(query, radius, indices, distance2);
				if (k < 0)
					throw std::runtime_error("search_radius_vector_xd() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "radius"_a)
		.def("search_hybrid_vector_xd", [](const KDTreeFlann &tree,
				const Eigen::VectorXd &query, double radius, int max_nn) {
				std::vector<int> indices; std::vector<double> distance2;
				int k = tree.SearchHybrid(query, radius, max_nn, indices,
						distance2);
				if (k < 0)
					throw std::runtime_error("search_hybrid_vector_xd() error!");
				return std::make_tuple(k, indices, distance2);
			}, "query"_a, "radius"_a, "max_nn"_a);
}
