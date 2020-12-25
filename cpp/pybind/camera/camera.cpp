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

#include "pybind/camera/camera.h"

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/camera/PinholeCameraTrajectory.h"
#include "pybind/docstring.h"

namespace open3d {
namespace camera {

void pybind_camera_classes(py::module &m) {
    // open3d.camera.PinholeCameraIntrinsic
    py::class_<PinholeCameraIntrinsic> pinhole_intr(
            m, "PinholeCameraIntrinsic",
            "PinholeCameraIntrinsic class stores intrinsic camera matrix, and "
            "image height and width.");
    py::detail::bind_default_constructor<PinholeCameraIntrinsic>(pinhole_intr);
    py::detail::bind_copy_functions<PinholeCameraIntrinsic>(pinhole_intr);
    pinhole_intr
            .def(py::init([](int w, int h, double fx, double fy, double cx,
                             double cy) {
                     return new PinholeCameraIntrinsic(w, h, fx, fy, cx, cy);
                 }),
                 "width"_a, "height"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a)
            .def(py::init([](PinholeCameraIntrinsicParameters param) {
                     return new PinholeCameraIntrinsic(param);
                 }),
                 "param"_a);
    pinhole_intr
            .def("set_intrinsics", &PinholeCameraIntrinsic::SetIntrinsics,
                 "width"_a, "height"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a,
                 "Set camera intrinsic parameters.")
            .def("get_focal_length", &PinholeCameraIntrinsic::GetFocalLength,
                 "Returns the focal length in a tuple of X-axis and Y-axis"
                 "focal lengths.")
            .def("get_principal_point",
                 &PinholeCameraIntrinsic::GetPrincipalPoint,
                 "Returns the principle point in a tuple of X-axis and."
                 "Y-axis principle points")
            .def("get_skew", &PinholeCameraIntrinsic::GetSkew,
                 "Returns the skew.")
            .def("is_valid", &PinholeCameraIntrinsic::IsValid,
                 "Returns True iff both the width and height are greater than "
                 "0.")
            .def_readwrite("width", &PinholeCameraIntrinsic::width_,
                           "int: Width of the image.")
            .def_readwrite("height", &PinholeCameraIntrinsic::height_,
                           "int: Height of the image.")
            .def_readwrite("intrinsic_matrix",
                           &PinholeCameraIntrinsic::intrinsic_matrix_,
                           "3x3 numpy array: Intrinsic camera matrix ``[[fx, "
                           "0, cx], [0, fy, "
                           "cy], [0, 0, 1]]``")
            .def("__repr__", [](const PinholeCameraIntrinsic &c) {
                return std::string("PinholeCameraIntrinsic with width = ") +
                       std::to_string(c.width_) +
                       std::string(" and height = ") +
                       std::to_string(c.height_) +
                       std::string(
                               ".\nAccess intrinsics with intrinsic_matrix.");
            });
    docstring::ClassMethodDocInject(m, "PinholeCameraIntrinsic", "__init__");
    docstring::ClassMethodDocInject(m, "PinholeCameraIntrinsic",
                                    "set_intrinsics",
                                    {{"width", "Width of the image."},
                                     {"height", "Height of the image."},
                                     {"fx", "X-axis focal length"},
                                     {"fy", "Y-axis focal length."},
                                     {"cx", "X-axis principle point."},
                                     {"cy", "Y-axis principle point."}});
    docstring::ClassMethodDocInject(m, "PinholeCameraIntrinsic",
                                    "get_focal_length");
    docstring::ClassMethodDocInject(m, "PinholeCameraIntrinsic",
                                    "get_principal_point");
    docstring::ClassMethodDocInject(m, "PinholeCameraIntrinsic", "get_skew");
    docstring::ClassMethodDocInject(m, "PinholeCameraIntrinsic", "is_valid");

    // open3d.camera.PinholeCameraIntrinsicParameters
    py::enum_<PinholeCameraIntrinsicParameters> pinhole_intr_params(
            m, "PinholeCameraIntrinsicParameters", py::arithmetic(),
            "PinholeCameraIntrinsicParameters");
    pinhole_intr_params
            .value("PrimeSenseDefault",
                   PinholeCameraIntrinsicParameters::PrimeSenseDefault,
                   "Default camera intrinsic parameter for PrimeSense.")
            .value("Kinect2DepthCameraDefault",
                   PinholeCameraIntrinsicParameters::Kinect2DepthCameraDefault,
                   "Default camera intrinsic parameter for Kinect2 depth "
                   "camera.")
            .value("Kinect2ColorCameraDefault",
                   PinholeCameraIntrinsicParameters::Kinect2ColorCameraDefault,
                   "Default camera intrinsic parameter for Kinect2 color "
                   "camera.")
            .export_values();
    pinhole_intr_params.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class that contains default camera intrinsic "
                       "parameters for different sensors.";
            }),
            py::none(), py::none(), "");

    // open3d.camera.PinholeCameraParameters
    py::class_<PinholeCameraParameters> pinhole_param(
            m, "PinholeCameraParameters",
            "Contains both intrinsic and extrinsic pinhole camera parameters.");
    py::detail::bind_default_constructor<PinholeCameraParameters>(
            pinhole_param);
    py::detail::bind_copy_functions<PinholeCameraParameters>(pinhole_param);
    pinhole_param
            .def_readwrite("intrinsic", &PinholeCameraParameters::intrinsic_,
                           "``open3d.camera.PinholeCameraIntrinsic``: "
                           "PinholeCameraIntrinsic "
                           "object.")
            .def_readwrite("extrinsic", &PinholeCameraParameters::extrinsic_,
                           "4x4 numpy array: Camera extrinsic parameters.")
            .def("__repr__", [](const PinholeCameraParameters &c) {
                return std::string("PinholeCameraParameters class.\n") +
                       std::string(
                               "Access its data via intrinsic and extrinsic.");
            });

    // open3d.camera.PinholeCameraTrajectory
    py::class_<PinholeCameraTrajectory> pinhole_traj(
            m, "PinholeCameraTrajectory",
            "Contains a list of ``PinholeCameraParameters``, useful to storing "
            "trajectories.");
    py::detail::bind_default_constructor<PinholeCameraTrajectory>(pinhole_traj);
    py::detail::bind_copy_functions<PinholeCameraTrajectory>(pinhole_traj);
    pinhole_traj
            .def_readwrite("parameters", &PinholeCameraTrajectory::parameters_,
                           "``List(open3d.camera.PinholeCameraParameters)``: "
                           "List of PinholeCameraParameters objects.")
            .def("__repr__", [](const PinholeCameraTrajectory &c) {
                return std::string("PinholeCameraTrajectory class.\n") +
                       std::string("Access its data via camera.parameters.");
            });
}

void pybind_camera(py::module &m) {
    py::module m_submodule = m.def_submodule("camera");
    pybind_camera_classes(m_submodule);
}

}  // namespace camera
}  // namespace open3d
