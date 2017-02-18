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

void pybind_image(py::module &m)
{
	/*
	py::class_<PointCloud, PyGeometry3D<PointCloud>,
			std::shared_ptr<PointCloud>, Geometry3D> pointcloud(m,
			"PointCloud");
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
	*/
}

void pybind_image_methods(py::module &m)
{
	/*
	m.def("CreatePointCloudFromFile", &CreatePointCloudFromFile,
			"Factory function to create a pointcloud from a file",
			"filename"_a);
	m.def("SelectDownSample", [](const PointCloud &input,
			const std::vector<size_t> &indices) {
			PointCloud output;
			if (SelectDownSample(input, indices, output) == false)
				throw std::runtime_error("SelectDownSample() error!");
			return output;
		}, "Function to select points from input pointcloud into output pointcloud",
			"input"_a, "indices"_a);
	*/
}
