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

#include "Open3D/Visualization/Visualizer/Visualizer.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithEditing.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithKeyCallback.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithVertexSelection.h"

#include "open3d_pybind/docstring.h"
#include "open3d_pybind/visualization/visualization.h"
#include "open3d_pybind/visualization/visualization_trampoline.h"

using namespace open3d;

// Functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_visualizer_docstrings = {
                {"callback_func", "The call back function."},
                {"depth_scale",
                 "Scale depth value when capturing the depth image."},
                {"do_render", "Set to ``True`` to do render."},
                {"filename", "Path to file."},
                {"geometry", "The ``Geometry`` object."},
                {"height", "Height of window."},
                {"left", "Left margin of the window to the screen."},
                {"top", "Top margin of the window to the screen."},
                {"visible", "Whether the window is visible."},
                {"width", "Width of the window."},
                {"window_name", "Window title name."},
                {"convert_to_world_coordinate",
                 "Set to ``True`` to convert to world coordinates"},
                {"reset_bounding_box",
                 "Set to ``False`` to keep current viewpoint"}};

void pybind_visualizer(py::module &m) {
    py::class_<visualization::Visualizer, PyVisualizer<>,
               std::shared_ptr<visualization::Visualizer>>
            visualizer(m, "Visualizer", "The main Visualizer class.");
    py::detail::bind_default_constructor<visualization::Visualizer>(visualizer);
    visualizer
            .def("__repr__",
                 [](const visualization::Visualizer &vis) {
                     return std::string("Visualizer with name ") +
                            vis.GetWindowName();
                 })
            .def("create_window",
                 &visualization::Visualizer::CreateVisualizerWindow,
                 "Function to create a window and initialize GLFW",
                 "window_name"_a = "Open3D", "width"_a = 1920,
                 "height"_a = 1080, "left"_a = 50, "top"_a = 50,
                 "visible"_a = true)
            .def("destroy_window",
                 &visualization::Visualizer::DestroyVisualizerWindow,
                 "Function to destroy a window. This function MUST be called "
                 "from the main thread.")
            .def("register_animation_callback",
                 &visualization::Visualizer::RegisterAnimationCallback,
                 "Function to register a callback function for animation. The "
                 "callback function returns if UpdateGeometry() needs to be "
                 "run.",
                 "callback_func"_a)
            .def("run", &visualization::Visualizer::Run,
                 "Function to activate the window. This function will block "
                 "the current thread until the window is closed.")
            .def("close", &visualization::Visualizer::Close,
                 "Function to notify the window to be closed")
            .def("reset_view_point", &visualization::Visualizer::ResetViewPoint,
                 "Function to reset view point")
            .def("update_geometry", &visualization::Visualizer::UpdateGeometry,
                 "Function to update geometry. This function must be called "
                 "when geometry has been changed. Otherwise the behavior of "
                 "Visualizer is undefined.",
                 "geometry"_a)
            .def("update_renderer", &visualization::Visualizer::UpdateRender,
                 "Function to inform render needed to be updated")
            .def("poll_events", &visualization::Visualizer::PollEvents,
                 "Function to poll events")
            .def("add_geometry", &visualization::Visualizer::AddGeometry,
                 "Function to add geometry to the scene and create "
                 "corresponding shaders",
                 "geometry"_a, "reset_bounding_box"_a = true)
            .def("remove_geometry", &visualization::Visualizer::RemoveGeometry,
                 "Function to remove geometry", "geometry"_a,
                 "reset_bounding_box"_a = true)
            .def("clear_geometries",
                 &visualization::Visualizer::ClearGeometries,
                 "Function to clear geometries from the visualizer")
            .def("get_view_control", &visualization::Visualizer::GetViewControl,
                 "Function to retrieve the associated ``ViewControl``",
                 py::return_value_policy::reference_internal)
            .def("get_render_option",
                 &visualization::Visualizer::GetRenderOption,
                 "Function to retrieve the associated ``RenderOption``",
                 py::return_value_policy::reference_internal)
            .def("capture_screen_float_buffer",
                 &visualization::Visualizer::CaptureScreenFloatBuffer,
                 "Function to capture screen and store RGB in a float buffer",
                 "do_render"_a = false)
            .def("capture_screen_image",
                 &visualization::Visualizer::CaptureScreenImage,
                 "Function to capture and save a screen image", "filename"_a,
                 "do_render"_a = false)
            .def("capture_depth_float_buffer",
                 &visualization::Visualizer::CaptureDepthFloatBuffer,
                 "Function to capture depth in a float buffer",
                 "do_render"_a = false)
            .def("capture_depth_image",
                 &visualization::Visualizer::CaptureDepthImage,
                 "Function to capture and save a depth image", "filename"_a,
                 "do_render"_a = false, "depth_scale"_a = 1000.0)
            .def("capture_depth_point_cloud",
                 &visualization::Visualizer::CaptureDepthPointCloud,
                 "Function to capture and save local point cloud", "filename"_a,
                 "do_render"_a = false, "convert_to_world_coordinate"_a = false)
            .def("get_window_name", &visualization::Visualizer::GetWindowName);

    py::class_<visualization::VisualizerWithKeyCallback,
               PyVisualizer<visualization::VisualizerWithKeyCallback>,
               std::shared_ptr<visualization::VisualizerWithKeyCallback>>
            visualizer_key(m, "VisualizerWithKeyCallback", visualizer,
                           "Visualizer with custom key callack capabilities.");
    py::detail::bind_default_constructor<
            visualization::VisualizerWithKeyCallback>(visualizer_key);
    visualizer_key
            .def("__repr__",
                 [](const visualization::VisualizerWithKeyCallback &vis) {
                     return std::string(
                                    "VisualizerWithKeyCallback with name ") +
                            vis.GetWindowName();
                 })
            .def("register_key_callback",
                 &visualization::VisualizerWithKeyCallback::RegisterKeyCallback,
                 "Function to register a callback function for a key press "
                 "event",
                 "key"_a, "callback_func"_a)

            .def("register_key_action_callback",
                 &visualization::VisualizerWithKeyCallback::
                         RegisterKeyActionCallback,
                 "Function to register a callback function for a key action "
                 "event. The callback function takes Visualizer, action and "
                 "mods as input and returns a boolean indicating if "
                 "UpdateGeometry() needs to be run.",
                 "key"_a, "callback_func"_a);

    py::class_<visualization::VisualizerWithEditing,
               PyVisualizer<visualization::VisualizerWithEditing>,
               std::shared_ptr<visualization::VisualizerWithEditing>>
            visualizer_edit(m, "VisualizerWithEditing", visualizer,
                            "Visualizer with editing capabilities.");
    py::detail::bind_default_constructor<visualization::VisualizerWithEditing>(
            visualizer_edit);
    visualizer_edit.def(py::init<double, bool, const std::string &>())
            .def("__repr__",
                 [](const visualization::VisualizerWithEditing &vis) {
                     return std::string("VisualizerWithEditing with name ") +
                            vis.GetWindowName();
                 })
            .def("get_picked_points",
                 &visualization::VisualizerWithEditing::GetPickedPoints,
                 "Function to get picked points");

    py::class_<visualization::VisualizerWithVertexSelection,
               PyVisualizer<visualization::VisualizerWithVertexSelection>,
               std::shared_ptr<visualization::VisualizerWithVertexSelection>>
            visualizer_vselect(
                    m, "VisualizerWithVertexSelection", visualizer,
                    "Visualizer with vertex selection capabilities.");
    py::detail::bind_default_constructor<
            visualization::VisualizerWithVertexSelection>(visualizer_vselect);
    visualizer_vselect.def(py::init<>())
            .def("__repr__",
                 [](const visualization::VisualizerWithVertexSelection &vis) {
                     return std::string(
                                    "VisualizerWithVertexSelection with "
                                    "name ") +
                            vis.GetWindowName();
                 })
            .def("get_picked_points",
                 &visualization::VisualizerWithVertexSelection::GetPickedPoints,
                 "Function to get picked points")
            .def("clear_picked_points",
                 &visualization::VisualizerWithVertexSelection::
                         ClearPickedPoints,
                 "Function to clear picked points")
            .def("register_selection_changed_callback",
                 &visualization::VisualizerWithVertexSelection::
                         RegisterSelectionChangedCallback,
                 "Registers a function to be called when selection changes")
            .def("register_selection_moving_callback",
                 &visualization::VisualizerWithVertexSelection::
                         RegisterSelectionMovingCallback,
                 "Registers a function to be called while selection moves. "
                 "Geometry's vertex values can be changed, but do not change"
                 "the number of vertices.")
            .def("register_selection_moved_callback",
                 &visualization::VisualizerWithVertexSelection::
                         RegisterSelectionMovedCallback,
                 "Registers a function to be called after selection moves");

    py::class_<visualization::VisualizerWithVertexSelection::PickedPoint>
            visualizer_vselect_pickedpoint(m, "PickedPoint");
    visualizer_vselect_pickedpoint.def(py::init<>())
            .def_readwrite("index",
                           &visualization::VisualizerWithVertexSelection::
                                   PickedPoint::index)
            .def_readwrite("coord",
                           &visualization::VisualizerWithVertexSelection::
                                   PickedPoint::coord);

    docstring::ClassMethodDocInject(m, "Visualizer", "add_geometry",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "remove_geometry",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer",
                                    "capture_depth_float_buffer",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "capture_depth_image",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer",
                                    "capture_depth_point_cloud",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer",
                                    "capture_screen_float_buffer",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "capture_screen_image",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "close",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "create_window",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "destroy_window",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "get_render_option",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "get_view_control",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "get_window_name",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "poll_events",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer",
                                    "register_animation_callback",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "reset_view_point",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "run",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "update_geometry",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "update_renderer",
                                    map_visualizer_docstrings);
}

void pybind_visualizer_method(py::module &m) {}
