// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/RenderOption.h"

#include "open3d/io/IJsonConvertibleIO.h"
#include "pybind/docstring.h"
#include "pybind/visualization/visualization.h"
#include "pybind/visualization/visualization_trampoline.h"

namespace open3d {
namespace visualization {

void pybind_renderoption_declarations(py::module &m) {
    py::class_<RenderOption, std::shared_ptr<RenderOption>> renderoption(
            m, "RenderOption", "Defines rendering options for visualizer.");
    // This is a nested class, but now it's bind to the module
    // o3d.visualization.PointColorOption
    py::native_enum<RenderOption::PointColorOption>(
            m, "PointColorOption", "enum.Enum",
            "Enum class for point color for ``PointCloud``.")
            .value("Default", RenderOption::PointColorOption::Default)
            .value("Color", RenderOption::PointColorOption::Color)
            .value("XCoordinate", RenderOption::PointColorOption::XCoordinate)
            .value("YCoordinate", RenderOption::PointColorOption::YCoordinate)
            .value("ZCoordinate", RenderOption::PointColorOption::ZCoordinate)
            .value("Normal", RenderOption::PointColorOption::Normal)
            .finalize();
    // This is a nested class, but now it's bind to the module
    // o3d.visualization.MeshShadeOption
    py::native_enum<RenderOption::MeshShadeOption>(
            m, "MeshShadeOption", "enum.Enum",
            "Enum class for mesh shading for ``TriangleMesh``.")
            .value("Default", RenderOption::MeshShadeOption::FlatShade)
            .value("Color", RenderOption::MeshShadeOption::SmoothShade)
            .finalize();

    // This is a nested class, but now it's bind to the module
    // o3d.visualization.MeshColorOption
    py::native_enum<RenderOption::MeshColorOption>(
            m, "MeshColorOption", "enum.Enum",
            "Enum class for color for ``TriangleMesh``.")
            .value("Default", RenderOption::MeshColorOption::Default)
            .value("Color", RenderOption::MeshColorOption::Color)
            .value("XCoordinate", RenderOption::MeshColorOption::XCoordinate)
            .value("YCoordinate", RenderOption::MeshColorOption::YCoordinate)
            .value("ZCoordinate", RenderOption::MeshColorOption::ZCoordinate)
            .value("Normal", RenderOption::MeshColorOption::Normal)
            .finalize();
}
void pybind_renderoption_definitions(py::module &m) {
    // open3d.visualization.RenderOption
    auto renderoption = static_cast<
            py::class_<RenderOption, std::shared_ptr<RenderOption>>>(
            m.attr("RenderOption"));
    py::detail::bind_default_constructor<RenderOption>(renderoption);
    renderoption
            .def("__repr__",
                 [](const RenderOption &vc) {
                     return std::string("RenderOption");
                 })
            .def(
                    "load_from_json",
                    [](RenderOption &ro, const fs::path &filename) {
                        io::ReadIJsonConvertible(filename.string(), ro);
                    },
                    "Function to load RenderOption from a JSON "
                    "file.",
                    "filename"_a)
            .def(
                    "save_to_json",
                    [](RenderOption &ro, const fs::path &filename) {
                        io::WriteIJsonConvertible(filename.string(), ro);
                    },
                    "Function to save RenderOption to a JSON "
                    "file.",
                    "filename"_a)
            .def_readwrite(
                    "background_color", &RenderOption::background_color_,
                    "float numpy array of size ``(3,)``: Background RGB color.")
            .def_readwrite("light_on", &RenderOption::light_on_,
                           "bool: Whether to turn on Phong lighting.")
            .def_readwrite("point_size", &RenderOption::point_size_,
                           "float: Point size for ``PointCloud``.")
            .def_readwrite("line_width", &RenderOption::line_width_,
                           "float: Line width for ``LineSet``.")
            .def_readwrite("point_show_normal",
                           &RenderOption::point_show_normal_,
                           "bool: Whether to show normal for ``PointCloud``.")
            .def_readwrite("show_coordinate_frame",
                           &RenderOption::show_coordinate_frame_,
                           "bool: Whether to show coordinate frame.")
            .def_readwrite(
                    "mesh_show_back_face", &RenderOption::mesh_show_back_face_,
                    "bool: Whether to show back faces for ``TriangleMesh``.")
            .def_readwrite(
                    "mesh_show_wireframe", &RenderOption::mesh_show_wireframe_,
                    "bool: Whether to show wireframe for ``TriangleMesh``.")
            .def_readwrite("point_color_option",
                           &RenderOption::point_color_option_,
                           "``PointColorOption``: Point color option for "
                           "``PointCloud``.")
            .def_readwrite("mesh_shade_option",
                           &RenderOption::mesh_shade_option_,
                           "``MeshShadeOption``: Mesh shading option for "
                           "``TriangleMesh``.")
            .def_readwrite(
                    "mesh_color_option", &RenderOption::mesh_color_option_,
                    "``MeshColorOption``: Color option for ``TriangleMesh``.");
    docstring::ClassMethodDocInject(m, "RenderOption", "load_from_json",
                                    {{"filename", "Path to file."}});
    docstring::ClassMethodDocInject(m, "RenderOption", "save_to_json",
                                    {{"filename", "Path to file."}});
}

}  // namespace visualization
}  // namespace open3d
