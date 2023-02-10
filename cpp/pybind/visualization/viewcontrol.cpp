// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/ViewControl.h"

#include "open3d/io/IJsonConvertibleIO.h"
#include "pybind/docstring.h"
#include "pybind/visualization/visualization.h"
#include "pybind/visualization/visualization_trampoline.h"

namespace open3d {
namespace visualization {

// Functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_view_control_docstrings = {
                {"parameter", "The pinhole camera parameter to convert from."},
                {"scale", "Scale ratio."},
                {"x", "Distance the mouse cursor has moved in x-axis."},
                {"y", "Distance the mouse cursor has moved in y-axis."},
                {"xo", "Original point coordinate of the mouse in x-axis."},
                {"yo", "Original point coordinate of the mouse in y-axis."},
                {"step", "The step to change field of view."},
                {"z_near", "The depth of the near z-plane of the visualizer."},
                {"z_far", "The depth of the far z-plane of the visualizer."},
};

void pybind_viewcontrol(py::module &m) {
    py::class_<ViewControl, PyViewControl<>, std::shared_ptr<ViewControl>>
            viewcontrol(m, "ViewControl", "View controller for visualizer.");
    py::detail::bind_default_constructor<ViewControl>(viewcontrol);
    viewcontrol
            .def("__repr__",
                 [](const ViewControl &vc) {
                     return std::string("ViewControl");
                 })
            .def(
                    "convert_to_pinhole_camera_parameters",
                    [](ViewControl &vc) {
                        camera::PinholeCameraParameters parameter;
                        vc.ConvertToPinholeCameraParameters(parameter);
                        return parameter;
                    },
                    "Function to convert ViewControl to "
                    "camera::PinholeCameraParameters")
            .def("convert_from_pinhole_camera_parameters",
                 &ViewControl::ConvertFromPinholeCameraParameters,
                 "parameter"_a, "allow_arbitrary"_a = false)
            .def("scale", &ViewControl::Scale, "Function to process scaling",
                 "scale"_a)
            .def("rotate", &ViewControl::Rotate, "Function to process rotation",
                 "x"_a, "y"_a, "xo"_a = 0.0, "yo"_a = 0.0)
            .def("translate", &ViewControl::Translate,
                 "Function to process translation", "x"_a, "y"_a, "xo"_a = 0.0,
                 "yo"_a = 0.0)
            .def("camera_local_translate", &ViewControl::CameraLocalTranslate,
                 "Function to process translation of camera", "forward"_a,
                 "right"_a, "up"_a)
            .def("camera_local_rotate", &ViewControl::CameraLocalRotate,
                 "Function to process rotation of camera in a local"
                 "coordinate frame",
                 "x"_a, "y"_a, "xo"_a = 0.0, "yo"_a = 0.0)
            .def("reset_camera_local_rotate",
                 &ViewControl::ResetCameraLocalRotate,
                 "Resets the coordinate frame for local camera rotations")
            .def("get_field_of_view", &ViewControl::GetFieldOfView,
                 "Function to get field of view")
            .def("change_field_of_view", &ViewControl::ChangeFieldOfView,
                 "Function to change field of view", "step"_a = 0.45)
            .def("set_constant_z_near", &ViewControl::SetConstantZNear,
                 "Function to change the near z-plane of the visualizer to a "
                 "constant value, i.e., independent of zoom and bounding box "
                 "size.",
                 "z_near"_a)
            .def("set_constant_z_far", &ViewControl::SetConstantZFar,
                 "Function to change the far z-plane of the visualizer to a "
                 "constant value, i.e., independent of zoom and bounding box "
                 "size.",
                 "z_far"_a)
            .def("unset_constant_z_near", &ViewControl::UnsetConstantZNear,
                 "Function to remove a previously set constant z near value, "
                 "i.e., near z-plane of the visualizer is dynamically set "
                 "dependent on zoom and bounding box size.")
            .def("unset_constant_z_far", &ViewControl::UnsetConstantZFar,
                 "Function to remove a previously set constant z far value, "
                 "i.e., far z-plane of the visualizer is dynamically set "
                 "dependent on zoom and bounding box size.")
            .def("set_lookat", &ViewControl::SetLookat,
                 "Set the lookat vector of the visualizer", "lookat"_a)
            .def("set_up", &ViewControl::SetUp,
                 "Set the up vector of the visualizer", "up"_a)
            .def("set_front", &ViewControl::SetFront,
                 "Set the front vector of the visualizer", "front"_a)
            .def("set_zoom", &ViewControl::SetZoom,
                 "Set the zoom of the visualizer", "zoom"_a);
    docstring::ClassMethodDocInject(m, "ViewControl", "change_field_of_view",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl",
                                    "convert_from_pinhole_camera_parameters",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl",
                                    "convert_to_pinhole_camera_parameters",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "get_field_of_view",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "rotate",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "scale",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "translate",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "set_constant_z_near",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "set_constant_z_far",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "unset_constant_z_near",
                                    map_view_control_docstrings);
    docstring::ClassMethodDocInject(m, "ViewControl", "unset_constant_z_far",
                                    map_view_control_docstrings);
}

void pybind_viewcontrol_method(py::module &m) {}

}  // namespace visualization
}  // namespace open3d
