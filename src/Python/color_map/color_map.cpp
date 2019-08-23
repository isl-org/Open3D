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

#include "Python/color_map/color_map.h"

#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/ColorMap/ColorMapOptimization.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"
#include "Python/docstring.h"

using namespace open3d;

void pybind_color_map_classes(py::module &m) {
    py::class_<color_map::ColorMapOptimizationOption>
            color_map_optimization_option(
                    m, "ColorMapOptimizationOption",
                    "Defines options for color map optimization.");
    py::detail::bind_default_constructor<color_map::ColorMapOptimizationOption>(
            color_map_optimization_option);
    color_map_optimization_option
            .def_readwrite(
                    "non_rigid_camera_coordinate",
                    &color_map::ColorMapOptimizationOption::
                            non_rigid_camera_coordinate_,
                    "bool: (Default ``False``) Set to ``True`` to enable "
                    "non-rigid optimization (optimizing camera "
                    "extrinsic params and image "
                    "wrapping field for color assignment), set to "
                    "``False`` to only enable rigid optimization "
                    "(optimize camera extrinsic params).")
            .def_readwrite(
                    "number_of_vertical_anchors",
                    &color_map::ColorMapOptimizationOption::
                            number_of_vertical_anchors_,
                    "int: (Default ``16``) Number of vertical anchor points "
                    "for image wrapping field. The number of horizontal anchor "
                    "points is computed automatically based on the "
                    "number of vertical anchor points. This option is "
                    "only used when non-rigid optimization is enabled.")
            .def_readwrite(
                    "non_rigid_anchor_point_weight",
                    &color_map::ColorMapOptimizationOption::
                            non_rigid_anchor_point_weight_,
                    "float: (Default ``0.316``) Additional regularization "
                    "terms added to non-rigid regularization. A higher value "
                    "results gives more conservative updates. If the residual "
                    "error does not stably decrease, it is mainly because "
                    "images are being bended abruptly. In this case, consider "
                    "making iteration more conservative by increasing "
                    "the value. This option is only used when non-rigid "
                    "optimization is enabled.")
            .def_readwrite(
                    "maximum_iteration",
                    &color_map::ColorMapOptimizationOption::maximum_iteration_,
                    "int: (Default ``300``) Number of iterations for "
                    "optimization steps.")
            .def_readwrite(
                    "maximum_allowable_depth",
                    &color_map::ColorMapOptimizationOption::
                            maximum_allowable_depth_,
                    "float: (Default ``2.5``) Parameter for point visibility "
                    "check. Points with depth larger than "
                    "``maximum_allowable_depth`` in a RGB-D will be marked as "
                    "invisible for the camera producing that RGB-D image. "
                    "Select a proper value to include necessary points while "
                    "ignoring unwanted points such as the background.")
            .def_readwrite(
                    "depth_threshold_for_visiblity_check",
                    &color_map::ColorMapOptimizationOption::
                            depth_threshold_for_visiblity_check_,
                    "float: (Default ``0.03``) Parameter for point visibility "
                    "check. When the difference of a point's depth "
                    "value in the RGB-D image and the point's depth "
                    "value in the 3D mesh is greater than "
                    "``depth_threshold_for_visiblity_check``, the "
                    "point is mark as invisible to the camera producing "
                    "the RGB-D image.")
            .def_readwrite(
                    "depth_threshold_for_discontinuity_check",
                    &color_map::ColorMapOptimizationOption::
                            depth_threshold_for_discontinuity_check_,
                    "float: (Default ``0.1``) Parameter for point visibility "
                    "check. It's often desirable to ignore points where there "
                    "are abrupt change in depth value. First the depth "
                    "gradient image is computed, points are considered "
                    "to be invisible if the depth gradient magnitude is "
                    "larger than ``depth_threshold_for_discontinuity_check``.")
            .def_readwrite(
                    "half_dilation_kernel_size_for_discontinuity_map",
                    &color_map::ColorMapOptimizationOption::
                            half_dilation_kernel_size_for_discontinuity_map_,
                    "int: (Default ``3``) Parameter for point visibility "
                    "check. Related to "
                    "``depth_threshold_for_discontinuity_check``, "
                    "when boundary points are detected, dilation is performed "
                    "to ignore points near the object boundary. "
                    "``half_dilation_kernel_size_for_discontinuity_map`` "
                    "specifies the half-kernel size for the dilation applied "
                    "on the visibility mask image.")
            .def_readwrite(
                    "image_boundary_margin",
                    &color_map::ColorMapOptimizationOption::
                            image_boundary_margin_,
                    "int: (Default ``10``) If a projected 3D point onto a 2D "
                    "image lies in the image border within "
                    "``image_boundary_margin``, the 3D point is "
                    "cosidered invisible from the camera producing the "
                    "image. This parmeter is not used for visibility "
                    "check, but used when computing the final color "
                    "assignment after color map optimization.")
            .def_readwrite(
                    "invisible_vertex_color_knn",
                    &color_map::ColorMapOptimizationOption::
                            invisible_vertex_color_knn_,
                    "int: (Default ``3``) If a vertex is invisible from all "
                    "images, we assign the averged color of the k nearest "
                    "visible vertices to fill the invisible vertex. Set to "
                    "``0`` to disable this feature and all invisible vertices "
                    "will be black.")
            .def("__repr__", [](const color_map::ColorMapOptimizationOption
                                        &to) {
                // clang-format off
                return fmt::format(
                    "color_map::ColorMapOptimizationOption with\n"
                    "- non_rigid_camera_coordinate: {}\n"
                    "- number_of_vertical_anchors: {}\n"
                    "- non_rigid_anchor_point_weight: {}\n"
                    "- maximum_iteration: {}\n"
                    "- maximum_allowable_depth: {}\n"
                    "- depth_threshold_for_visiblity_check: {}\n"
                    "- depth_threshold_for_discontinuity_check: {}\n"
                    "- half_dilation_kernel_size_for_discontinuity_map: {}\n"
                    "- image_boundary_margin: {}\n"
                    "- invisible_vertex_color_knn: {}\n",
                    to.non_rigid_camera_coordinate_,
                    to.number_of_vertical_anchors_,
                    to.non_rigid_anchor_point_weight_,
                    to.maximum_iteration_,
                    to.maximum_allowable_depth_,
                    to.depth_threshold_for_visiblity_check_,
                    to.depth_threshold_for_discontinuity_check_,
                    to.half_dilation_kernel_size_for_discontinuity_map_,
                    to.image_boundary_margin_,
                    to.invisible_vertex_color_knn_
                );
                // clang-format on
            });
}

void pybind_color_map_methods(py::module &m) {
    m.def("color_map_optimization", &color_map::ColorMapOptimization,
          "Function for color mapping of reconstructed scenes via optimization",
          "mesh"_a, "imgs_rgbd"_a, "camera"_a,
          "option"_a = color_map::ColorMapOptimizationOption());
    docstring::FunctionDocInject(
            m, "color_map_optimization",
            {{"mesh", "The input geometry mesh."},
             {"imgs_rgbd", "A list of RGBD images seen by cameras."},
             {"camera", "Cameras' parameters."},
             {"option", "The ColorMap optimization option."}});
}

void pybind_color_map(py::module &m) {
    py::module m_submodule = m.def_submodule("color_map");
    pybind_color_map_classes(m_submodule);
    pybind_color_map_methods(m_submodule);
}
