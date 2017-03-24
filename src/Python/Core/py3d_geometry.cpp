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
#include "py3d_core_trampoline.h"

#include <Core/Core.h>
using namespace three;

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
			std::shared_ptr<Geometry3D>, Geometry> geometry3d(m, "Geometry3D");
	geometry3d
		.def("GetMinBound", &Geometry3D::GetMinBound)
		.def("GetMaxBound", &Geometry3D::GetMaxBound)
		.def("Transform", &Geometry3D::Transform);
	
	py::class_<Geometry2D, PyGeometry2D<Geometry2D>,
			std::shared_ptr<Geometry2D>, Geometry> geometry2d(m, "Geometry2D");
	geometry2d
		.def("GetMinBound", &Geometry2D::GetMinBound)
		.def("GetMaxBound", &Geometry2D::GetMaxBound);
}
