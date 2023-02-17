// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "open3d/visualization/utility/SelectionPolygonVolume.h"
#include "open3d/visualization/visualizer/Visualizer.h"
#include "pybind/docstring.h"
#include "pybind/visualization/visualization.h"

namespace open3d {
namespace visualization {

void pybind_visualization_utility(py::module &m) {
    py::class_<SelectionPolygonVolume> selection_volume(
            m, "SelectionPolygonVolume",
            "Select a polygon volume for cropping.");
    py::detail::bind_default_constructor<SelectionPolygonVolume>(
            selection_volume);
    py::detail::bind_copy_functions<SelectionPolygonVolume>(selection_volume);
    selection_volume
            .def(
                    "crop_point_cloud",
                    [](const SelectionPolygonVolume &s,
                       const geometry::PointCloud &input) {
                        return s.CropPointCloud(input);
                    },
                    "input"_a, "Function to crop point cloud.")
            .def(
                    "crop_triangle_mesh",
                    [](const SelectionPolygonVolume &s,
                       const geometry::TriangleMesh &input) {
                        return s.CropTriangleMesh(input);
                    },
                    "input"_a, "Function to crop crop triangle mesh.")
            .def(
                    "crop_in_polygon",
                    [](const SelectionPolygonVolume &s,
                       const geometry::PointCloud &input) {
                        return s.CropInPolygon(input);
                    },
                    "input"_a, "Function to crop 3d point clouds.")
            .def("__repr__",
                 [](const SelectionPolygonVolume &s) {
                     return std::string(
                             "SelectionPolygonVolume, access "
                             "its members:\n"
                             "orthogonal_axis, bounding_polygon, axis_min, "
                             "axis_max");
                 })
            .def_readwrite("orthogonal_axis",
                           &SelectionPolygonVolume::orthogonal_axis_,
                           "string: one of ``{x, y, z}``.")
            .def_readwrite("bounding_polygon",
                           &SelectionPolygonVolume::bounding_polygon_,
                           "``(n, 3)`` float64 numpy array: Bounding polygon "
                           "boundary.")
            .def_readwrite("axis_min", &SelectionPolygonVolume::axis_min_,
                           "float: Minimum axis value.")
            .def_readwrite("axis_max", &SelectionPolygonVolume::axis_max_,
                           "float: Maximum axis value.");
    docstring::ClassMethodDocInject(m, "SelectionPolygonVolume",
                                    "crop_point_cloud",
                                    {{"input", "The input point cloud."}});
    docstring::ClassMethodDocInject(m, "SelectionPolygonVolume",
                                    "crop_triangle_mesh",
                                    {{"input", "The input triangle mesh."}});
    docstring::ClassMethodDocInject(m, "SelectionPolygonVolume",
                                    "crop_in_polygon",
                                    {{"input", "The input point cloud xyz."}});
}

// Visualization util functions have similar arguments, sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"callback_function",
                 "Call back function to be triggered at a key press event."},
                {"filename", "The file path."},
                {"geometry_list", "List of geometries to be visualized."},
                {"height", "The height of the visualization window."},
                {"key_to_callback", "Map of key to call back functions."},
                {"left", "The left margin of the visualization window."},
                {"optional_view_trajectory_json_file",
                 "Camera trajectory json file path for custom animation."},
                {"top", "The top margin of the visualization window."},
                {"width", "The width of the visualization window."},
                {"point_show_normal",
                 "Visualize point normals if set to true."},
                {"mesh_show_wireframe",
                 "Visualize mesh wireframe if set to true."},
                {"mesh_show_back_face",
                 "Visualize also the back face of the mesh triangles."},
                {"window_name",
                 "The displayed title of the visualization window."},
                {"lookat", "The lookat vector of the camera."},
                {"up", "The up vector of the camera."},
                {"front", "The front vector of the camera."},
                {"zoom", "The zoom of the camera."}};

void pybind_visualization_utility_methods(py::module &m) {
    m.def(
            "draw_geometries",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::string &window_name, int width, int height, int left,
               int top, bool point_show_normal, bool mesh_show_wireframe,
               bool mesh_show_back_face) {
                std::string current_dir =
                        utility::filesystem::GetWorkingDirectory();
                DrawGeometries(geometry_ptrs, window_name, width, height, left,
                               top, point_show_normal, mesh_show_wireframe,
                               mesh_show_back_face);
                utility::filesystem::ChangeWorkingDirectory(current_dir);
            },
            "Function to draw a list of geometry::Geometry objects",
            "geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
            "height"_a = 1080, "left"_a = 50, "top"_a = 50,
            "point_show_normal"_a = false, "mesh_show_wireframe"_a = false,
            "mesh_show_back_face"_a = false);
    m.def(
            "draw_geometries",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::string &window_name, int width, int height, int left,
               int top, bool point_show_normal, bool mesh_show_wireframe,
               bool mesh_show_back_face, Eigen::Vector3d lookat,
               Eigen::Vector3d up, Eigen::Vector3d front, double zoom) {
                std::string current_dir =
                        utility::filesystem::GetWorkingDirectory();
                DrawGeometries(geometry_ptrs, window_name, width, height, left,
                               top, point_show_normal, mesh_show_wireframe,
                               mesh_show_back_face, &lookat, &up, &front,
                               &zoom);
                utility::filesystem::ChangeWorkingDirectory(current_dir);
            },
            "Function to draw a list of geometry::Geometry objects",
            "geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
            "height"_a = 1080, "left"_a = 50, "top"_a = 50,
            "point_show_normal"_a = false, "mesh_show_wireframe"_a = false,
            "mesh_show_back_face"_a = false, "lookat"_a, "up"_a, "front"_a,
            "zoom"_a);
    docstring::FunctionDocInject(m, "draw_geometries",
                                 map_shared_argument_docstrings);

    m.def(
            "draw_geometries_with_custom_animation",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::string &window_name, int width, int height, int left,
               int top, const std::string &json_filename) {
                std::string current_dir =
                        utility::filesystem::GetWorkingDirectory();
                DrawGeometriesWithCustomAnimation(geometry_ptrs, window_name,
                                                  width, height, left, top,
                                                  json_filename);
                utility::filesystem::ChangeWorkingDirectory(current_dir);
            },
            "Function to draw a list of geometry::Geometry objects with a GUI "
            "that "
            "supports animation",
            "geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
            "height"_a = 1080, "left"_a = 50, "top"_a = 50,
            "optional_view_trajectory_json_file"_a = "");
    docstring::FunctionDocInject(m, "draw_geometries_with_custom_animation",
                                 map_shared_argument_docstrings);

    m.def(
            "draw_geometries_with_animation_callback",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               std::function<bool(Visualizer *)> callback_func,
               const std::string &window_name, int width, int height, int left,
               int top) {
                std::string current_dir =
                        utility::filesystem::GetWorkingDirectory();
                DrawGeometriesWithAnimationCallback(geometry_ptrs,
                                                    callback_func, window_name,
                                                    width, height, left, top);
                utility::filesystem::ChangeWorkingDirectory(current_dir);
            },
            "Function to draw a list of geometry::Geometry objects with a "
            "customized "
            "animation callback function",
            "geometry_list"_a, "callback_function"_a,
            "window_name"_a = "Open3D", "width"_a = 1920, "height"_a = 1080,
            "left"_a = 50, "top"_a = 50, py::return_value_policy::reference);
    docstring::FunctionDocInject(m, "draw_geometries_with_animation_callback",
                                 map_shared_argument_docstrings);

    m.def(
            "draw_geometries_with_key_callbacks",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::map<int, std::function<bool(Visualizer *)>>
                       &key_to_callback,
               const std::string &window_name, int width, int height, int left,
               int top) {
                std::string current_dir =
                        utility::filesystem::GetWorkingDirectory();
                DrawGeometriesWithKeyCallbacks(geometry_ptrs, key_to_callback,
                                               window_name, width, height, left,
                                               top);
                utility::filesystem::ChangeWorkingDirectory(current_dir);
            },
            "Function to draw a list of geometry::Geometry objects with a "
            "customized "
            "key-callback mapping",
            "geometry_list"_a, "key_to_callback"_a, "window_name"_a = "Open3D",
            "width"_a = 1920, "height"_a = 1080, "left"_a = 50, "top"_a = 50);
    docstring::FunctionDocInject(m, "draw_geometries_with_key_callbacks",
                                 map_shared_argument_docstrings);

    m.def(
            "draw_geometries_with_editing",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::string &window_name, int width, int height, int left,
               int top) {
                DrawGeometriesWithEditing(geometry_ptrs, window_name, width,
                                          height, left, top);
            },
            "Function to draw a list of geometry::Geometry providing user "
            "interaction",
            "geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
            "height"_a = 1080, "left"_a = 50, "top"_a = 50);
    docstring::FunctionDocInject(m, "draw_geometries_with_editing",
                                 map_shared_argument_docstrings);

    m.def(
            "draw_geometries_with_vertex_selection",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::string &window_name, int width, int height, int left,
               int top) {
                DrawGeometriesWithVertexSelection(geometry_ptrs, window_name,
                                                  width, height, left, top);
            },
            "Function to draw a list of geometry::Geometry providing ability "
            "for user to select points",
            "geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
            "height"_a = 1080, "left"_a = 50, "top"_a = 50);
    docstring::FunctionDocInject(m, "draw_geometries_with_vertex_selection",
                                 map_shared_argument_docstrings);

    m.def(
            "read_selection_polygon_volume",
            [](const std::string &filename) {
                SelectionPolygonVolume vol;
                io::ReadIJsonConvertible(filename, vol);
                return vol;
            },
            "Function to read SelectionPolygonVolume from file", "filename"_a);
    docstring::FunctionDocInject(m, "read_selection_polygon_volume",
                                 map_shared_argument_docstrings);
}

}  // namespace visualization
}  // namespace open3d
