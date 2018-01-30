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

#include "py3d_visualization.h"

#include <Core/Utility/FileSystem.h>
#include <Core/Geometry/PointCloud.h>
#include <Visualization/Utility/SelectionPolygonVolume.h>
#include <Visualization/Utility/DrawGeometry.h>
#include <Visualization/Visualizer/Visualizer.h>
#include <IO/ClassIO/IJsonConvertibleIO.h>
using namespace three;

void pybind_utility(py::module &m)
{
	py::class_<SelectionPolygonVolume> selection_volume(m,
			"SelectionPolygonVolume");
	py::detail::bind_default_constructor<SelectionPolygonVolume>(
			selection_volume);
	py::detail::bind_copy_functions<SelectionPolygonVolume>(selection_volume);
	selection_volume
		.def("crop_point_cloud", [](const SelectionPolygonVolume &s,
				const PointCloud &input) {
			return s.CropPointCloud(input);
		}, "input"_a)
		.def("__repr__", [](const SelectionPolygonVolume &s) {
			return std::string("SelectionPolygonVolume, access its members:\n"
					"orthogonal_axis, bounding_polygon, axis_min, axis_max");
		})
		.def_readwrite("orthogonal_axis",
				&SelectionPolygonVolume::orthogonal_axis_)
		.def_readwrite("bounding_polygon",
				&SelectionPolygonVolume::bounding_polygon_)
		.def_readwrite("axis_min", &SelectionPolygonVolume::axis_min_)
		.def_readwrite("axis_max", &SelectionPolygonVolume::axis_max_);
}

void pybind_utility_methods(py::module &m)
{
	m.def("draw_geometries",
	[](const std::vector<std::shared_ptr<const Geometry>> &geometry_ptrs,
			const std::string &window_name, int width, int height,
			int left, int top) {
		std::string current_dir = filesystem::GetWorkingDirectory();
		DrawGeometries(geometry_ptrs, window_name, width, height, left, top);
		filesystem::ChangeWorkingDirectory(current_dir);
	}, "Function to draw a list of Geometry objects",
			"geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
			"height"_a = 1080, "left"_a = 50, "top"_a = 50);
	m.def("draw_geometries_with_custom_animation",
	[](const std::vector<std::shared_ptr<const Geometry>> &geometry_ptrs,
			const std::string &window_name, int width, int height,
			int left, int top, const std::string &json_filename) {
		std::string current_dir = filesystem::GetWorkingDirectory();
		DrawGeometriesWithCustomAnimation(geometry_ptrs, window_name, width,
				height, left, top, json_filename);
		filesystem::ChangeWorkingDirectory(current_dir);
	}, "Function to draw a list of Geometry objects with a GUI that supports animation",
			"geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
			"height"_a = 1080, "left"_a = 50, "top"_a = 50,
			"optional_view_trajectory_json_file"_a = "");
	m.def("draw_geometries_with_animation_callback",
	[](const std::vector<std::shared_ptr<const Geometry>> &geometry_ptrs,
			std::function<bool(Visualizer *)> callback_func,
			const std::string &window_name, int width, int height,
			int left, int top) {
		std::string current_dir = filesystem::GetWorkingDirectory();
		DrawGeometriesWithAnimationCallback(geometry_ptrs, callback_func,
				window_name, width, height, left, top);
		filesystem::ChangeWorkingDirectory(current_dir);
	}, "Function to draw a list of Geometry objects with a customized animation callback function",
			"geometry_list"_a, "callback_function"_a,
			"window_name"_a = "Open3D", "width"_a = 1920,
			"height"_a = 1080, "left"_a = 50, "top"_a = 50,
			py::return_value_policy::reference);
	m.def("draw_geometries_with_key_callbacks",
	[](const std::vector<std::shared_ptr<const Geometry>> &geometry_ptrs,
			const std::map<int, std::function<bool(Visualizer *)>>
			&key_to_callback, const std::string &window_name, int width,
			int height, int left, int top) {
		std::string current_dir = filesystem::GetWorkingDirectory();
		DrawGeometriesWithKeyCallbacks(geometry_ptrs, key_to_callback,
				window_name, width, height, left, top);
		filesystem::ChangeWorkingDirectory(current_dir);
	}, "Function to draw a list of Geometry objects with a customized key-callback mapping",
			"geometry_list"_a, "key_to_callback"_a, "window_name"_a = "Open3D",
			"width"_a = 1920, "height"_a = 1080, "left"_a = 50, "top"_a = 50);
	m.def("read_selection_polygon_volume", [](const std::string &filename) {
		SelectionPolygonVolume vol;
		ReadIJsonConvertible(filename, vol);
		return vol;
	}, "Function to read SelectionPolygonVolume from file", "filename"_a);
}
