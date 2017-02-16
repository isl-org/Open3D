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

#include <Core/Geometry/Geometry.h>
#include <Core/Geometry/Geometry2D.h>
#include <Core/Geometry/Geometry3D.h>
#include <Core/Geometry/PointCloud.h>
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
		.def("__repr__", [](PointCloud &pcd) {
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

	m.def("CreatePointCloudFromFile", &CreatePointCloudFromFile,
			"Factory function to create a pointcloud from a file",
			"filename"_a);
	m.def("VoxelDownSample", &VoxelDownSample,
			"Function to downsample input pointcloud into output pointcloud with a voxel",
			"input"_a, "voxel_size"_a, "output"_a);
}
