// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <Core/Core.h>
using namespace three;


template <class GeometryBase = Geometry> class PyGeometry : public GeometryBase
{
public:
	using GeometryBase::GeometryBase;
	void Clear() override { PYBIND11_OVERLOAD_PURE(void, GeometryBase, ); }
	bool IsEmpty() const override {
		PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
	}
};

template <class Geometry3DBase = Geometry3D> class PyGeometry3D :
		public PyGeometry<Geometry3DBase>
{
public:
	using PyGeometry<Geometry3DBase>::PyGeometry;
	Eigen::Vector3d GetMinBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
	}
	Eigen::Vector3d GetMaxBound() const override {
		PYBIND11_OVERLOAD_PURE(Eigen::Vector3d, Geometry3DBase, );
	}
	void Transform(const Eigen::Matrix4d &transformation) override {
		PYBIND11_OVERLOAD_PURE(void, Geometry3DBase, transformation);
	}
};

void pybind_geometry(py::module &m)
{
	py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
			geometry(m, "Geometry");
	geometry
		.def("Clear", &Geometry::Clear)
		.def("IsEmpty", &Geometry::IsEmpty)
		.def("GetGeometryType", &Geometry::GetGeometryType)
		.def("Dimension", &Geometry::Dimension);
	py::enum_<Geometry::GeometryType>(geometry, "Type", py::arithmetic())
		.value("Unspecified", Geometry::GEOMETRY_UNSPECIFIED)
		.value("PointCloud", Geometry::GEOMETRY_POINTCLOUD)
		.value("LineSet", Geometry::GEOMETRY_LINESET)
		.value("TriangleMesh", Geometry::GEOMETRY_TRIANGLEMESH)
		.value("Image", Geometry::GEOMETRY_IMAGE)
		.export_values();

	py::class_<Geometry3D, PyGeometry3D<Geometry3D>,
			std::shared_ptr<Geometry3D>> geometry3d(m, "Geometry3D", geometry);
	geometry3d
		.def("GetMinBound", &Geometry3D::GetMinBound)
		.def("GetMaxBound", &Geometry3D::GetMaxBound)
		.def("Transform", &Geometry3D::Transform);

	py::class_<PointCloud, PyGeometry3D<PointCloud>,
			std::shared_ptr<PointCloud>> pointcloud(m, "PointCloud",
			geometry3d);
	pointcloud.def(py::init<>())
		.def("__repr__", [](const PointCloud &pcd) {
			return std::string("PointCloud with ") + 
					std::to_string(pcd.points_.size()) + " points.";
		})
		.def("HasPoints", &PointCloud::HasPoints)
		.def("HasNormals", &PointCloud::HasNormals)
		.def("HasColors", &PointCloud::HasColors)
		.def("NormalizeNormals", &PointCloud::NormalizeNormals)
		.def_readwrite("points", &PointCloud::points_)
		.def_readwrite("normals", &PointCloud::normals_)
		.def_readwrite("colors", &PointCloud::colors_);

	py::class_<KDTreeSearchParam> kdtreesearchparam(m, "KDTreeSearchParam");
	kdtreesearchparam
		.def("GetSearchType", &KDTreeSearchParam::GetSearchType);
	py::enum_<KDTreeSearchParam::SearchType>(kdtreesearchparam, "Type",
			py::arithmetic())
		.value("KNNSearch", KDTreeSearchParam::SEARCH_KNN)
		.value("RadiusSearch", KDTreeSearchParam::SEARCH_RADIUS)
		.value("HybridSearch", KDTreeSearchParam::SEARCH_HYBRID)
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
		.def("__repr__", [](const KDTreeSearchParamRadius &param){
			return std::string("KDTreeSearchParamRadius with radius = ") +
					std::to_string(param.radius_);
		})
		.def_readwrite("radius", &KDTreeSearchParamRadius::radius_);

	py::class_<KDTreeSearchParamHybrid> kdtreesearchparam_hybrid(m,
			"KDTreeSearchParamHybrid", kdtreesearchparam);
	kdtreesearchparam_hybrid
		.def(py::init<double, int>(), "radius"_a, "max_nn"_a)
		.def("__repr__", [](const KDTreeSearchParamHybrid &param){
			return std::string("KDTreeSearchParamHybrid with radius = ") +
					std::to_string(param.radius_) + " and max_nn = " +
					std::to_string(param.max_nn_);
		})
		.def_readwrite("radius", &KDTreeSearchParamHybrid::radius_)
		.def_readwrite("max_nn", &KDTreeSearchParamHybrid::max_nn_);

	py::class_<KDTreeFlann> kdtreeflann(m, "KDTreeFlann");
	kdtreeflann.def(py::init<>())
		.def(py::init<const Geometry &>(), "geometry"_a)
		.def("SetGeometry", &KDTreeFlann::SetGeometry, "geometry"_a)
		.def("SearchVector3D", &KDTreeFlann::Search<Eigen::Vector3d>,
				"query"_a, "search_param"_a, "indices"_a, "distance2"_a)
		.def("SearchKNNVector3D", &KDTreeFlann::SearchKNN<Eigen::Vector3d>,
				"query"_a, "knn"_a, "indices"_a, "distance2"_a)
		.def("SearchRadiusVector3D",
				&KDTreeFlann::SearchRadius<Eigen::Vector3d>, "query"_a,
				"radius"_a, "indices"_a, "distance2"_a)
		.def("SearchHybridVector3D",
				&KDTreeFlann::SearchHybrid<Eigen::Vector3d>, "query"_a,
				"radius"_a, "max_nn"_a, "indices"_a, "distance2"_a);

	m.def("CreatePointCloudFromFile", &CreatePointCloudFromFile,
			"Factory function to create a pointcloud from a file",
			"filename"_a);
	m.def("SelectDownSample", &SelectDownSample,
			"Function to select points from input pointcloud into output pointcloud",
			"input"_a, "indices"_a, "output"_a);
	m.def("VoxelDownSample", &VoxelDownSample,
			"Function to downsample input pointcloud into output pointcloud with a voxel",
			"input"_a, "voxel_size"_a, "output"_a);
	m.def("UniformDownSample", &UniformDownSample,
			"Function to downsample input pointcloud into output pointcloud uniformly",
			"input"_a, "every_k_points"_a, "output"_a);
	m.def("CropPointCloud", &CropPointCloud,
			"input"_a, "min_bound"_a, "max_bound"_a, "output"_a);
}
