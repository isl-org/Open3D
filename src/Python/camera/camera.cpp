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

#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <Open3D/Camera/PinholeCameraTrajectory.h>
#include <Open3D/IO/ClassIO/IJsonConvertibleIO.h>
#include <Open3D/IO/ClassIO/PinholeCameraTrajectoryIO.h>

#include "Python/camera/camera.h"

using namespace open3d;

void pybind_camera_classes(py::module &m) {
    py::class_<camera::PinholeCameraIntrinsic> pinhole_intr(
            m, "PinholeCameraIntrinsic", "PinholeCameraIntrinsic");
    py::detail::bind_default_constructor<camera::PinholeCameraIntrinsic>(
            pinhole_intr);
    py::detail::bind_copy_functions<camera::PinholeCameraIntrinsic>(
            pinhole_intr);
    pinhole_intr
            .def(py::init([](int w, int h, double fx, double fy, double cx,
                             double cy) {
                     return new camera::PinholeCameraIntrinsic(w, h, fx, fy, cx,
                                                               cy);
                 }),
                 "width"_a, "height"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a)
            .def(py::init([](camera::PinholeCameraIntrinsicParameters param) {
                     return new camera::PinholeCameraIntrinsic(param);
                 }),
                 "param"_a)
            .def("set_intrinsics",
                 &camera::PinholeCameraIntrinsic::SetIntrinsics, "width"_a,
                 "height"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a)
            .def("get_focal_length",
                 &camera::PinholeCameraIntrinsic::GetFocalLength)
            .def("get_principal_point",
                 &camera::PinholeCameraIntrinsic::GetPrincipalPoint)
            .def("get_skew", &camera::PinholeCameraIntrinsic::GetSkew)
            .def("is_valid", &camera::PinholeCameraIntrinsic::IsValid)
            .def_readwrite("width", &camera::PinholeCameraIntrinsic::width_)
            .def_readwrite("height", &camera::PinholeCameraIntrinsic::height_)
            .def_readwrite("intrinsic_matrix",
                           &camera::PinholeCameraIntrinsic::intrinsic_matrix_)
            .def("__repr__", [](const camera::PinholeCameraIntrinsic &c) {
                return std::string(
                               "camera::PinholeCameraIntrinsic with width = ") +
                       std::to_string(c.width_) +
                       std::string(" and height = ") +
                       std::to_string(c.height_) +
                       std::string(
                               ".\nAccess intrinsics with intrinsic_matrix.");
            });
    py::enum_<camera::PinholeCameraIntrinsicParameters>(
            m, "PinholeCameraIntrinsicParameters", py::arithmetic(),
            "PinholeCameraIntrinsicParameters")
            .value("PrimeSenseDefault",
                   camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault)
            .value("Kinect2DepthCameraDefault",
                   camera::PinholeCameraIntrinsicParameters::
                           Kinect2DepthCameraDefault)
            .value("Kinect2ColorCameraDefault",
                   camera::PinholeCameraIntrinsicParameters::
                           Kinect2ColorCameraDefault)
            .export_values();

    py::class_<camera::PinholeCameraParameters> pinhole_param(
            m, "PinholeCameraParameters", "PinholeCameraParameters");
    py::detail::bind_default_constructor<camera::PinholeCameraParameters>(
            pinhole_param);
    py::detail::bind_copy_functions<camera::PinholeCameraParameters>(
            pinhole_param);
    pinhole_param
            .def_readwrite("intrinsic",
                           &camera::PinholeCameraParameters::intrinsic_)
            .def_readwrite("extrinsic",
                           &camera::PinholeCameraParameters::extrinsic_)
            .def("__repr__", [](const camera::PinholeCameraParameters &c) {
                return std::string("camera::PinholeCameraParameters class.\n") +
                       std::string(
                               "Access its data via intrinsic and extrinsic.");
            });

    py::class_<camera::PinholeCameraTrajectory> pinhole_traj(
            m, "PinholeCameraTrajectory", "PinholeCameraTrajectory");
    py::detail::bind_default_constructor<camera::PinholeCameraTrajectory>(
            pinhole_traj);
    py::detail::bind_copy_functions<camera::PinholeCameraTrajectory>(
            pinhole_traj);
    pinhole_traj
            .def_readwrite("parameters",
                           &camera::PinholeCameraTrajectory::parameters_)
            .def("__repr__", [](const camera::PinholeCameraTrajectory &c) {
                return std::string("camera::PinholeCameraTrajectory class.\n") +
                       std::string("Access its data via camera_parameters.");
            });
}

void pybind_camera_methods(py::module &m) {
    m.def("read_pinhole_camera_intrinsic",
          [](const std::string &filename) {
              camera::PinholeCameraIntrinsic intrinsic;
              io::ReadIJsonConvertible(filename, intrinsic);
              return intrinsic;
          },
          "Function to read camera::PinholeCameraIntrinsic from file",
          "filename"_a);
    m.def("write_pinhole_camera_intrinsic",
          [](const std::string &filename,
             const camera::PinholeCameraIntrinsic &intrinsic) {
              return io::WriteIJsonConvertible(filename, intrinsic);
          },
          "Function to write camera::PinholeCameraIntrinsic to file",
          "filename"_a, "intrinsic"_a);
    m.def("read_pinhole_camera_parameters",
          [](const std::string &filename) {
              camera::PinholeCameraParameters parameters;
              io::ReadIJsonConvertible(filename, parameters);
              return parameters;
          },
          "Function to read camera::PinholeCameraParameters from file",
          "filename"_a);
    m.def("write_pinhole_camera_parameters",
          [](const std::string &filename,
             const camera::PinholeCameraParameters &parameters) {
              return io::WriteIJsonConvertible(filename, parameters);
          },
          "Function to write camera::PinholeCameraParameters to file",
          "filename"_a, "parameters"_a);
    m.def("read_pinhole_camera_trajectory",
          [](const std::string &filename) {
              camera::PinholeCameraTrajectory trajectory;
              io::ReadPinholeCameraTrajectory(filename, trajectory);
              return trajectory;
          },
          "Function to read camera::PinholeCameraTrajectory from file",
          "filename"_a);
    m.def("write_pinhole_camera_trajectory",
          [](const std::string &filename,
             const camera::PinholeCameraTrajectory &trajectory) {
              return io::WritePinholeCameraTrajectory(filename, trajectory);
          },
          "Function to write camera::PinholeCameraTrajectory to file",
          "filename"_a, "trajectory"_a);
}

void pybind_camera(py::module &m) {
    py::module m_submodule = m.def_submodule("camera");
    pybind_camera_classes(m_submodule);
    pybind_camera_methods(m_submodule);
}
