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

#include <Core/Core.h>
using namespace three;

void pybind_geometry(py::module &m)
{
	py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
			geometry(m, "Geometry");
	geometry
		.def("clear", &Geometry::Clear)
		.def("is_empty", &Geometry::IsEmpty)
		.def("get_geometry_type", &Geometry::GetGeometryType)
		.def("dimension", &Geometry::Dimension);
	py::enum_<Geometry::GeometryType>(geometry, "Type", py::arithmetic())
		.value("Unspecified", Geometry::GeometryType::Unspecified)
		.value("PointCloud", Geometry::GeometryType::PointCloud)
		.value("LineSet", Geometry::GeometryType::LineSet)
		.value("TriangleMesh", Geometry::GeometryType::TriangleMesh)
		.value("Image", Geometry::GeometryType::Image)
		.export_values();

	py::class_<Geometry3D, PyGeometry3D<Geometry3D>,
			std::shared_ptr<Geometry3D>, Geometry> geometry3d(m, "Geometry3D");
	geometry3d
		.def("get_min_bound", &Geometry3D::GetMinBound)
		.def("get_max_bound", &Geometry3D::GetMaxBound)
		.def("transform", &Geometry3D::Transform);

	py::class_<Geometry2D, PyGeometry2D<Geometry2D>,
			std::shared_ptr<Geometry2D>, Geometry> geometry2d(m, "Geometry2D");
	geometry2d
		.def("get_min_bound", &Geometry2D::GetMinBound)
		.def("get_max_bound", &Geometry2D::GetMaxBound);
}
