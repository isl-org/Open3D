// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/camera/camera.h"

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/camera/PinholeCameraTrajectory.h"
#include "pybind/docstring.h"

namespace open3d {
namespace camera {

void pybind_camera_declarations(py::module &m) {
    py::module m_camera = m.def_submodule("camera");
    py::class_<PinholeCameraIntrinsic> pinhole_intr(
            m_camera, "PinholeCameraIntrinsic",
            "PinholeCameraIntrinsic class stores intrinsic camera matrix, and "
            "image height and width.");
    // open3d.camera.PinholeCameraIntrinsicParameters
    py::native_enum<PinholeCameraIntrinsicParameters> pinhole_intr_params(
            m_camera, "PinholeCameraIntrinsicParameters", "enum.Enum",
            "Enum class that contains default camera intrinsic parameters for "
            "different sensors.");
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
            .export_values()
            .finalize();
    py::class_<PinholeCameraParameters> pinhole_param(
            m_camera, "PinholeCameraParameters",
            "Contains both intrinsic and extrinsic pinhole camera parameters.");
    py::class_<PinholeCameraTrajectory> pinhole_traj(
            m_camera, "PinholeCameraTrajectory",
            "Contains a list of ``PinholeCameraParameters``, useful to storing "
            "trajectories.");
}
void pybind_camera_definitions(py::module &m) {
    auto m_camera = static_cast<py::module>(m.attr("camera"));
    // open3d.camera.PinholeCameraIntrinsic
    auto pinhole_intr = static_cast<py::class_<PinholeCameraIntrinsic>>(
            m_camera.attr("PinholeCameraIntrinsic"));
    py::detail::bind_default_constructor<PinholeCameraIntrinsic>(pinhole_intr);
    py::detail::bind_copy_functions<PinholeCameraIntrinsic>(pinhole_intr);
    pinhole_intr
            .def(py::init<int, int, const Eigen::Matrix3d>(), "width"_a,
                 "height"_a, "intrinsic_matrix"_a)
            .def(py::init<int, int, double, double, double, double>(),
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
                return fmt::format(
                        "PinholeCameraIntrinsic("
                        "width={}, "
                        "height={}, "
                        ")",
                        c.width_, c.height_);
            });
    docstring::ClassMethodDocInject(m_camera, "PinholeCameraIntrinsic",
                                    "__init__");
    docstring::ClassMethodDocInject(m_camera, "PinholeCameraIntrinsic",
                                    "set_intrinsics",
                                    {{"width", "Width of the image."},
                                     {"height", "Height of the image."},
                                     {"fx", "X-axis focal length"},
                                     {"fy", "Y-axis focal length."},
                                     {"cx", "X-axis principle point."},
                                     {"cy", "Y-axis principle point."}});
    docstring::ClassMethodDocInject(m_camera, "PinholeCameraIntrinsic",
                                    "get_focal_length");
    docstring::ClassMethodDocInject(m_camera, "PinholeCameraIntrinsic",
                                    "get_principal_point");
    docstring::ClassMethodDocInject(m_camera, "PinholeCameraIntrinsic",
                                    "get_skew");
    docstring::ClassMethodDocInject(m_camera, "PinholeCameraIntrinsic",
                                    "is_valid");

    // open3d.camera.PinholeCameraParameters
    auto pinhole_param = static_cast<py::class_<PinholeCameraParameters>>(
            m_camera.attr("PinholeCameraParameters"));
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
    auto pinhole_traj = static_cast<py::class_<PinholeCameraTrajectory>>(
            m_camera.attr("PinholeCameraTrajectory"));
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

}  // namespace camera
}  // namespace open3d
