// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/Visualizer.h"

#include "open3d/geometry/Image.h"
#include "open3d/visualization/visualizer/VisualizerWithEditing.h"
#include "open3d/visualization/visualizer/VisualizerWithKeyCallback.h"
#include "open3d/visualization/visualizer/VisualizerWithVertexSelection.h"
#include "pybind/docstring.h"
#include "pybind/visualization/visualization.h"
#include "pybind/visualization/visualization_trampoline.h"

namespace open3d {
namespace visualization {

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

void pybind_visualizer_declarations(py::module &m) {
    py::class_<Visualizer, PyVisualizer<>, std::shared_ptr<Visualizer>>
            visualizer(m, "Visualizer", "The main Visualizer class.");
    py::class_<VisualizerWithKeyCallback,
               PyVisualizer<VisualizerWithKeyCallback>,
               std::shared_ptr<VisualizerWithKeyCallback>>
            visualizer_key(m, "VisualizerWithKeyCallback", visualizer,
                           "Visualizer with custom key callback capabilities.");
    py::class_<VisualizerWithEditing, PyVisualizer<VisualizerWithEditing>,
               std::shared_ptr<VisualizerWithEditing>>
            visualizer_edit(m, "VisualizerWithEditing", visualizer,
                            "Visualizer with editing capabilities.");
    py::class_<VisualizerWithVertexSelection,
               PyVisualizer<VisualizerWithVertexSelection>,
               std::shared_ptr<VisualizerWithVertexSelection>>
            visualizer_vselect(
                    m, "VisualizerWithVertexSelection", visualizer,
                    "Visualizer with vertex selection capabilities.");
    py::class_<VisualizerWithVertexSelection::PickedPoint>
            visualizer_vselect_pickedpoint(m, "PickedPoint");
}

void pybind_visualizer_definitions(py::module &m) {
    auto visualizer = static_cast<py::class_<Visualizer, PyVisualizer<>,
                                             std::shared_ptr<Visualizer>>>(
            m.attr("Visualizer"));
    py::detail::bind_default_constructor<Visualizer>(visualizer);
    visualizer
            .def("__repr__",
                 [](const Visualizer &vis) {
                     return std::string("Visualizer with name ") +
                            vis.GetWindowName();
                 })
            .def("create_window", &Visualizer::CreateVisualizerWindow,
                 "Function to create a window and initialize GLFW",
                 "window_name"_a = "Open3D", "width"_a = 1920,
                 "height"_a = 1080, "left"_a = 50, "top"_a = 50,
                 "visible"_a = true)
            .def("destroy_window", &Visualizer::DestroyVisualizerWindow,
                 "Function to destroy a window. This function MUST be called "
                 "from the main thread.")
            .def("register_animation_callback",
                 &Visualizer::RegisterAnimationCallback,
                 "Function to register a callback function for animation. The "
                 "callback function returns if UpdateGeometry() needs to be "
                 "run.",
                 "callback_func"_a)
            .def("run", &Visualizer::Run,
                 "Function to activate the window. This function will block "
                 "the current thread until the window is closed.")
            .def("close", &Visualizer::Close,
                 "Function to notify the window to be closed")
            .def("reset_view_point", &Visualizer::ResetViewPoint,
                 "Function to reset view point", "reset_bounding_box"_a = false)
            .def("update_geometry", &Visualizer::UpdateGeometry,
                 "Function to update geometry. This function must be called "
                 "when geometry has been changed. Otherwise the behavior of "
                 "Visualizer is undefined.",
                 "geometry"_a)
            .def("update_renderer", &Visualizer::UpdateRender,
                 "Function to inform render needed to be updated")
            .def("set_full_screen", &Visualizer::SetFullScreen,
                 "Function to change between fullscreen and windowed",
                 "fullscreen"_a)
            .def("toggle_full_screen", &Visualizer::ToggleFullScreen,
                 "Function to toggle between fullscreen and windowed")
            .def("is_full_screen", &Visualizer::IsFullScreen,
                 "Function to query whether in fullscreen mode")
            .def("poll_events", &Visualizer::PollEvents,
                 "Function to poll events")
            .def("add_geometry", &Visualizer::AddGeometry,
                 "Function to add geometry to the scene and create "
                 "corresponding shaders",
                 "geometry"_a, "reset_bounding_box"_a = true)
            .def("remove_geometry", &Visualizer::RemoveGeometry,
                 "Function to remove geometry", "geometry"_a,
                 "reset_bounding_box"_a = true)
            .def("clear_geometries", &Visualizer::ClearGeometries,
                 "Function to clear geometries from the visualizer")
            .def("get_view_control", &Visualizer::GetViewControl,
                 "Function to retrieve the associated ``ViewControl``",
                 py::return_value_policy::reference_internal)
            .def("get_render_option", &Visualizer::GetRenderOption,
                 "Function to retrieve the associated ``RenderOption``",
                 py::return_value_policy::reference_internal)
            .def("capture_screen_float_buffer",
                 &Visualizer::CaptureScreenFloatBuffer,
                 "Function to capture screen and store RGB in a float buffer",
                 "do_render"_a = false)
            .def(
                    "capture_screen_image",
                    [](Visualizer &self, const fs::path &filename,
                       bool do_render) {
                        return self.CaptureScreenImage(filename.string(),
                                                       do_render);
                    },
                    "Function to capture and save a screen image", "filename"_a,
                    "do_render"_a = false)
            .def("capture_depth_float_buffer",
                 &Visualizer::CaptureDepthFloatBuffer,
                 "Function to capture depth in a float buffer",
                 "do_render"_a = false)
            .def(
                    "capture_depth_image",
                    [](Visualizer &self, const fs::path &filename,
                       bool do_render, double depth_scale) {
                        self.CaptureDepthImage(filename.string(), do_render,
                                               depth_scale);
                    },
                    "Function to capture and save a depth image", "filename"_a,
                    "do_render"_a = false, "depth_scale"_a = 1000.0)
            .def(
                    "capture_depth_point_cloud",
                    [](Visualizer &self, const fs::path &filename,
                       bool do_render, bool convert_to_world_coordinate) {
                        self.CaptureDepthPointCloud(
                                filename.string(), do_render,
                                convert_to_world_coordinate);
                    },
                    "Function to capture and save local point cloud",
                    "filename"_a, "do_render"_a = false,
                    "convert_to_world_coordinate"_a = false)
            .def("get_window_name", &Visualizer::GetWindowName)
            .def("get_view_status", &Visualizer::GetViewStatus,
                 "Get the current view status as a json string of "
                 "ViewTrajectory.")
            .def("set_view_status", &Visualizer::SetViewStatus,
                 "Set the current view status from a json string of "
                 "ViewTrajectory.",
                 "view_status_str"_a);
    auto visualizer_key =
            static_cast<py::class_<VisualizerWithKeyCallback,
                                   PyVisualizer<VisualizerWithKeyCallback>,
                                   std::shared_ptr<VisualizerWithKeyCallback>>>(
                    m.attr("VisualizerWithKeyCallback"));
    py::detail::bind_default_constructor<VisualizerWithKeyCallback>(
            visualizer_key);
    visualizer_key
            .def("__repr__",
                 [](const VisualizerWithKeyCallback &vis) {
                     return std::string(
                                    "VisualizerWithKeyCallback with name ") +
                            vis.GetWindowName();
                 })
            .def("register_key_callback",
                 &VisualizerWithKeyCallback::RegisterKeyCallback,
                 "Function to register a callback function for a key press "
                 "event",
                 "key"_a, "callback_func"_a)

            .def("register_key_action_callback",
                 &VisualizerWithKeyCallback::RegisterKeyActionCallback,
                 "Function to register a callback function for a key action "
                 "event. The callback function takes `Visualizer`, `action` "
                 "and `mods` as input and returns a boolean indicating if "
                 "`UpdateGeometry()` needs to be run.  The `action` can be one "
                 "of `GLFW_RELEASE` (0), `GLFW_PRESS` (1) or `GLFW_REPEAT` "
                 "(2), see `GLFW input interface "
                 "<https://www.glfw.org/docs/latest/group__input.html>`__. The "
                 "`mods` specifies the modifier key, see `GLFW modifier key "
                 "<https://www.glfw.org/docs/latest/group__mods.html>`__",
                 "key"_a, "callback_func"_a)

            .def("register_mouse_move_callback",
                 &VisualizerWithKeyCallback::RegisterMouseMoveCallback,
                 "Function to register a callback function for a mouse move "
                 "event. The callback function takes Visualizer, x and y mouse "
                 "position inside the window as input and returns a boolean "
                 "indicating if UpdateGeometry() needs to be run. `GLFW mouse "
                 "position <https://www.glfw.org/docs/latest/"
                 "input_guide.html#input_mouse>`__ for more details.",
                 "callback_func"_a)

            .def("register_mouse_scroll_callback",
                 &VisualizerWithKeyCallback::RegisterMouseScrollCallback,
                 "Function to register a callback function for a mouse scroll "
                 "event. The callback function takes Visualizer, x and y mouse "
                 "scroll offset as input and returns a boolean "
                 "indicating if UpdateGeometry() needs to be run. `GLFW mouse "
                 "scrolling <https://www.glfw.org/docs/latest/"
                 "input_guide.html#scrolling>`__ for more details.",
                 "callback_func"_a)

            .def("register_mouse_button_callback",
                 &VisualizerWithKeyCallback::RegisterMouseButtonCallback,
                 "Function to register a callback function for a mouse button "
                 "event. The callback function takes `Visualizer`, `button`, "
                 "`action` and `mods` as input and returns a boolean "
                 "indicating `UpdateGeometry()` needs to be run. The `action` "
                 "can be one of GLFW_RELEASE (0), GLFW_PRESS (1) or "
                 "GLFW_REPEAT (2), see `GLFW input interface "
                 "<https://www.glfw.org/docs/latest/group__input.html>`__.  "
                 "The `mods` specifies the modifier key, see `GLFW modifier "
                 "key <https://www.glfw.org/docs/latest/group__mods.html>`__.",
                 "callback_func"_a);

    auto visualizer_edit =
            static_cast<py::class_<VisualizerWithEditing,
                                   PyVisualizer<VisualizerWithEditing>,
                                   std::shared_ptr<VisualizerWithEditing>>>(
                    m.attr("VisualizerWithEditing"));
    py::detail::bind_default_constructor<VisualizerWithEditing>(
            visualizer_edit);
    visualizer_edit
            .def(py::init<double, bool, const std::string &>(), "voxel_size"_a,
                 "use_dialog"_a, "directory"_a)
            .def("__repr__",
                 [](const VisualizerWithEditing &vis) {
                     return std::string("VisualizerWithEditing with name ") +
                            vis.GetWindowName();
                 })
            .def("get_picked_points", &VisualizerWithEditing::GetPickedPoints,
                 "Function to get picked points")
            .def("get_cropped_geometry",
                 &VisualizerWithEditing::GetCroppedGeometry,
                 "Function to get cropped geometry");

    auto visualizer_vselect = static_cast<
            py::class_<VisualizerWithVertexSelection,
                       PyVisualizer<VisualizerWithVertexSelection>,
                       std::shared_ptr<VisualizerWithVertexSelection>>>(
            m.attr("VisualizerWithVertexSelection"));
    py::detail::bind_default_constructor<VisualizerWithVertexSelection>(
            visualizer_vselect);
    visualizer_vselect.def(py::init<>())
            .def("__repr__",
                 [](const VisualizerWithVertexSelection &vis) {
                     return std::string(
                                    "VisualizerWithVertexSelection with "
                                    "name ") +
                            vis.GetWindowName();
                 })
            .def("pick_points", &VisualizerWithVertexSelection::PickPoints,
                 "Function to pick points", "x"_a, "y"_a, "w"_a, "h"_a)
            .def("get_picked_points",
                 &VisualizerWithVertexSelection::GetPickedPoints,
                 "Function to get picked points")
            .def("clear_picked_points",
                 &VisualizerWithVertexSelection::ClearPickedPoints,
                 "Function to clear picked points")
            .def("add_picked_points",
                 &VisualizerWithVertexSelection::AddPickedPoints,
                 "Function to add picked points", "indices"_a)
            .def("remove_picked_points",
                 &VisualizerWithVertexSelection::RemovePickedPoints,
                 "Function to remove picked points", "indices"_a)
            .def("register_selection_changed_callback",
                 &VisualizerWithVertexSelection::
                         RegisterSelectionChangedCallback,
                 "Registers a function to be called when selection changes",
                 "f"_a)
            .def("register_selection_moving_callback",
                 &VisualizerWithVertexSelection::
                         RegisterSelectionMovingCallback,
                 "Registers a function to be called while selection moves. "
                 "Geometry's vertex values can be changed, but do not change"
                 "the number of vertices.",
                 "f"_a)
            .def("register_selection_moved_callback",
                 &VisualizerWithVertexSelection::RegisterSelectionMovedCallback,
                 "Registers a function to be called after selection moves",
                 "f"_a);

    auto visualizer_vselect_pickedpoint =
            static_cast<py::class_<VisualizerWithVertexSelection::PickedPoint>>(
                    m.attr("PickedPoint"));
    visualizer_vselect_pickedpoint.def(py::init<>())
            .def_readwrite("index",
                           &VisualizerWithVertexSelection::PickedPoint::index)
            .def_readwrite("coord",
                           &VisualizerWithVertexSelection::PickedPoint::coord);

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
    docstring::ClassMethodDocInject(m, "Visualizer", "set_full_screen",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "toggle_full_screen",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "is_full_screen",
                                    map_visualizer_docstrings);
}

}  // namespace visualization
}  // namespace open3d
