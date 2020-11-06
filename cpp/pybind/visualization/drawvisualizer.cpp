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

#include <pybind/visualization/visualization.h>

#include <open3d/geometry/Geometry3D.h>
#include <open3d/t/geometry/Geometry.h>
#include <open3d/visualization/gui/Window.h>
#include <open3d/visualization/visualizer/DrawVisualizer.h>
#include <open3d/visualization/rendering/Open3DScene.h>

namespace open3d {
namespace visualization {

using namespace visualizer;
    
void pybind_drawvisualizer(py::module &m) {
    py::class_<DrawVisualizer, UnownedPointer<DrawVisualizer>, gui::Window> drawvis(m, "DrawVisualizer", "Visualization object used by draw()");

    py::enum_<DrawVisualizer::Shader> dv_shader(drawvis, "Shader",
                                                "Scene-level rendering options");
    dv_shader.value("STANDARD", DrawVisualizer::Shader::STANDARD,
                    "Pixel colors from standard lighting model")
             .value("NORMALS", DrawVisualizer::Shader::NORMALS,
                    "Pixel colors correspond to surface normal")
             .value("DEPTH", DrawVisualizer::Shader::DEPTH,
                    "Pixel colors correspond to depth buffer value")
             .export_values();

    py::class_<DrawVisualizer::DrawObject> drawobj(drawvis, "DrawObject",
                                                   "Information about an object that is drawn. Do not modify this, it can lead to unexpected results.");
    drawobj.def_readonly("name", &DrawVisualizer::DrawObject::name,
                         "The name of the object")
           .def_property_readonly("geometry",
                         [](const DrawVisualizer::DrawObject& o) {
                             if (o.geometry) {
                                 return py::cast(o.geometry);
                             } else {
                                 return py::cast(o.tgeometry);
                             }
                         },
                         "The geometry. Modifying this will not "
                         "result in any visible change. Use "
                         "remove_geometry() and then add_geometry()"
                         "to change the geometry")
           .def_readonly("group", &DrawVisualizer::DrawObject::group,
                         "The group that the object belongs to")
           .def_readonly("time", &DrawVisualizer::DrawObject::time,
                         "The object's timestamp")
           .def_readonly("is_visible",
                         &DrawVisualizer::DrawObject::is_visible,
                         "True if the object is checked in the list. "
                         "If the object's group is unchecked or an "
                         "animation is playing, the object's "
                         "visiblity may not correspond with this "
                         "value");

    drawvis.def(py::init<const std::string, int, int>(),
                "title"_a = "Open3D", "width"_a = 1024, "height"_a = 768,
                "Creates a DrawVisualizer object")
            .def("add_action", &DrawVisualizer::AddAction,
                 "Adds a button to the custom actions section of the UI. "
                 "add_action(name, callback). The callback will be given "
                 "one parameter, the DrawVisualizer instance, and does not "
                 "return any value.")
            .def("add_menu_action", &DrawVisualizer::AddMenuAction,
                 "Adds a menu item to the \"Actions\" Menu. "
                 "add_menu_action(name, callback). The callback will be given "
                 "one parameter, the DrawVisualizer instance, and does not "
                 "return any value.")
            .def("add_geometry",
                 py::overload_cast<const std::string&,
                                   std::shared_ptr<geometry::Geometry3D>,
                                   rendering::Material *,
                                   const std::string&,
                                   double,
                                   bool>(&DrawVisualizer::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a = nullptr, "group"_a = "",
                 "time"_a = 0.0, "is_visible"_a = true,
                 "Adds a geometry: geometry(name, geometry, material=None, group='', time=0.0, is_visible=True). 'name' must be unique.")
/*            .def("add_geometry",
                 py::overload_cast<const std::string&,
                                   std::shared_ptr<t::geometry::Geometry>,
                                   rendering::Material *,
                                   const std::string&,
                                   double,
                                   bool>(&DrawVisualizer::AddGeometry),
                 "Adds a geometry")
*/
            .def("add_geometry",
                 [](py::object dv, const py::dict& d) {
                     rendering::Material *material = nullptr;
                     std::string group = "";
                     double time = 0;
                     bool is_visible = true;

                     std::string name = py::cast<std::string>(d["name"]);
                     if (d.contains("material")) {
                         material = py::cast<rendering::Material*>(d["material"]);
                     }
                     if (d.contains("group")) {
                         group = py::cast<std::string>(d["group"]);
                     }
                     if (d.contains("time")) {
                         time = py::cast<double>(d["time"]);
                     }
                     if (d.contains("is_visible")) {
                         is_visible = py::cast<bool>(d["is_visible"]);
                     }
                     py::object g = d["geometry"];
                     // Instead of trying to figure out how to cast 'g' as the
                     // appropriate shared_ptr, if we can call the function using
                     // pybind and let it figure out how to do everything.
                     dv.attr("add_geometry")(name, g, material, group, time,
                                             is_visible);
                 },
                 "Adds a geometry from a dictionary. The dictionary has the "
                 "following elements:\n"
                 "name: unique name of the object (required)\n"
                 "geometry: the geometry or t.geometry object (required)\n"
                 "material: a visualization.rendering.Material object "
                 "(optional)\n"
                 "group: a string declaring the group it is a member of "
                 "(optional)\n"
                 "time: a time value\n")
            .def("remove_geometry", &DrawVisualizer::RemoveGeometry,
                 "remove_geometry(name): removes the geometry with the "
                 "name.")
            .def("show_geometry", &DrawVisualizer::ShowGeometry,
                 "Checks or unchecks the named geometry in the list. Note that "
                 "even if show_geometry(name, True) is called, the object may "
                 "not actually be visible if its group is unchecked, or if an "
                 "animation is in progress.")
            .def("get_geometry", &DrawVisualizer::GetGeometry,
                 "get_geometry(name): Returns the DrawObject corresponding to "
                 "the name. This should be treated as read-only. Modify "
                 "visibility with show_geometry(), and other values by "
                 "removing the object and re-adding it with the new values")
            .def("reset_camera", &DrawVisualizer::ResetCamera,
                 "Sets camera to default position")
            .def("export_current_image", &DrawVisualizer::ExportCurrentImage,
                 "export_image(path). Exports a PNG image of what is "
                 "currently displayed to the given path.")
            .def_property("show_settings", [](const DrawVisualizer& dv) {
                              return dv.GetUIState().show_settings;
                          },
                          &DrawVisualizer::ShowSettings,
                          "Gets/sets if settings panel is visible")
            .def_property("background_color",
                          [](const DrawVisualizer& dv) {
                              return dv.GetUIState().bg_color;
                          },
                          &DrawVisualizer::SetBackgroundColor,
                          "Gets/sets the background color")
            .def_property("scene_shader", 
                          [](const DrawVisualizer& dv) {
                              return dv.GetUIState().scene_shader;
                          },
                          &DrawVisualizer::SetShader,
                          "Gets/sets the shading model for the scene")
            .def_property("show_axes",
                          [](const DrawVisualizer& dv) {
                              return dv.GetUIState().show_axes;
                          },
                          &DrawVisualizer::ShowAxes,
                          "Gets/sets if axes are visible")
            .def_property_readonly("scene", &DrawVisualizer::GetScene,
                                   "Returns the rendering.Open3DScene object "
                                   "for low-level manipulation")
            .def_property("current_time", &DrawVisualizer::GetCurrentTime,
                          &DrawVisualizer::SetCurrentTime,
                          "Gets/sets the current time. If setting, only the "
                          "objects belonging to the current time-step will "
                          "be displayed")
            .def_property("animation_time_step",
                          &DrawVisualizer::GetAnimationTimeStep,
                          &DrawVisualizer::SetAnimationTimeStep,
                          "Gets/sets the time step for animations. Default is "
                          "1.0")
            .def_property("animation_frame_delay",
                          &DrawVisualizer::GetAnimationFrameDelay,
                          &DrawVisualizer::SetAnimationFrameDelay,
                          "Gets/sets the length of time a frame is visible.")
            .def_property("is_animating", &DrawVisualizer::GetIsAnimating,
                          &DrawVisualizer::SetAnimating,
                          "Gets/sets the status of the animation. Changing "
                          "value will start or stop the animating.")
        ;
}

}  // namespace open3d
}  // namespace visualization
