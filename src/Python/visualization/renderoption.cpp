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

#include "Open3D/Visualization/Visualizer/RenderOption.h"
#include "Open3D/IO/ClassIO/IJsonConvertibleIO.h"
#include "Python/docstring.h"
#include "Python/visualization/visualization.h"
#include "Python/visualization/visualization_trampoline.h"
using namespace open3d;

void pybind_renderoption(py::module &m) {
    // open3d.visualization.RenderOption
    py::class_<visualization::RenderOption,
               std::shared_ptr<visualization::RenderOption>>
            renderoption(m, "RenderOption",
                         "Defines rendering options for visulizer.");
    py::detail::bind_default_constructor<visualization::RenderOption>(
            renderoption);
    renderoption
            .def("__repr__",
                 [](const visualization::RenderOption &vc) {
                     return std::string("RenderOption");
                 })
            .def("load_from_json",
                 [](visualization::RenderOption &ro,
                    const std::string &filename) {
                     io::ReadIJsonConvertible(filename, ro);
                 },
                 "Function to load visualization::RenderOption from a JSON "
                 "file.",
                 "filename"_a)
            .def("save_to_json",
                 [](visualization::RenderOption &ro,
                    const std::string &filename) {
                     io::WriteIJsonConvertible(filename, ro);
                 },
                 "Function to save visualization::RenderOption to a JSON "
                 "file.",
                 "filename"_a)
            .def_readwrite(
                    "background_color",
                    &visualization::RenderOption::background_color_,
                    "float numpy array of size ``(3,)``: Background RGB color.")
            .def_readwrite("light_on", &visualization::RenderOption::light_on_,
                           "bool: Whether to turn on Phong lighting.")
            .def_readwrite("point_size",
                           &visualization::RenderOption::point_size_,
                           "float: Point size for ``PointCloud``.")
            .def_readwrite("line_width",
                           &visualization::RenderOption::line_width_,
                           "float: Line width for ``LineSet``.")
            .def_readwrite("point_show_normal",
                           &visualization::RenderOption::point_show_normal_,
                           "bool: Whether to show normal for ``PointCloud``.")
            .def_readwrite("show_coordinate_frame",
                           &visualization::RenderOption::show_coordinate_frame_,
                           "bool: Whether to show coordinate frame.")
            .def_readwrite(
                    "mesh_show_back_face",
                    &visualization::RenderOption::mesh_show_back_face_,
                    "bool: Whether to show back faces for ``TriangleMesh``.")
            .def_readwrite("point_color_option",
                           &visualization::RenderOption::point_color_option_,
                           "``PointColorOption``: Point color option for "
                           "``PointCloud``.")
            .def_readwrite("mesh_shade_option",
                           &visualization::RenderOption::mesh_shade_option_,
                           "``MeshShadeOption``: Mesh shading option for "
                           "``TriangleMesh``.")
            .def_readwrite(
                    "mesh_color_option",
                    &visualization::RenderOption::mesh_color_option_,
                    "``MeshColorOption``: Color option for ``TriangleMesh``.");
    docstring::ClassMethodDocInject(m, "RenderOption", "load_from_json",
                                    {{"filename", "Path to file."}});
    docstring::ClassMethodDocInject(m, "RenderOption", "save_to_json",
                                    {{"filename", "Path to file."}});

    // This is a nested class, but now it's bind to the module
    // o3d.visualization.PointColorOption
    py::enum_<visualization::RenderOption::PointColorOption>
            enum_point_color_option(m, "PointColorOption", py::arithmetic(),
                                    "PointColorOption");
    enum_point_color_option.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for point color for ``PointCloud``.";
            }),
            py::none(), py::none(), "");
    enum_point_color_option
            .value("Default",
                   visualization::RenderOption::PointColorOption::Default)
            .value("Color",
                   visualization::RenderOption::PointColorOption::Color)
            .value("XCoordinate",
                   visualization::RenderOption::PointColorOption::XCoordinate)
            .value("YCoordinate",
                   visualization::RenderOption::PointColorOption::YCoordinate)
            .value("ZCoordinate",
                   visualization::RenderOption::PointColorOption::ZCoordinate)
            .value("Normal",
                   visualization::RenderOption::PointColorOption::Normal)
            .export_values();

    // This is a nested class, but now it's bind to the module
    // o3d.visualization.MeshShadeOption
    py::enum_<visualization::RenderOption::MeshShadeOption>
            enum_mesh_shade_option(m, "MeshShadeOption", py::arithmetic(),
                                   "MeshShadeOption");
    enum_mesh_shade_option.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for msh shading for ``TriangleMesh``.";
            }),
            py::none(), py::none(), "");
    enum_mesh_shade_option
            .value("Default",
                   visualization::RenderOption::MeshShadeOption::FlatShade)
            .value("Color",
                   visualization::RenderOption::MeshShadeOption::SmoothShade)
            .export_values();

    // This is a nested class, but now it's bind to the module
    // o3d.visualization.MeshColorOption
    py::enum_<visualization::RenderOption::MeshColorOption>
            enum_mesh_clor_option(m, "MeshColorOption", py::arithmetic(),
                                  "MeshColorOption");
    enum_mesh_clor_option.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for color for ``TriangleMesh``.";
            }),
            py::none(), py::none(), "");
    enum_mesh_clor_option
            .value("Default",
                   visualization::RenderOption::MeshColorOption::Default)
            .value("Color", visualization::RenderOption::MeshColorOption::Color)
            .value("XCoordinate",
                   visualization::RenderOption::MeshColorOption::XCoordinate)
            .value("YCoordinate",
                   visualization::RenderOption::MeshColorOption::YCoordinate)
            .value("ZCoordinate",
                   visualization::RenderOption::MeshColorOption::ZCoordinate)
            .value("Normal",
                   visualization::RenderOption::MeshColorOption::Normal)
            .export_values();
}

void pybind_renderoption_method(py::module &m) {}
