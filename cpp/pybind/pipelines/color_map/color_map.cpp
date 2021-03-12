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

#include "pybind/pipelines/color_map/color_map.h"

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/color_map/NonRigidOptimizer.h"
#include "open3d/pipelines/color_map/RigidOptimizer.h"
#include "open3d/utility/Console.h"
#include "pybind/docstring.h"

namespace open3d {
namespace pipelines {
namespace color_map {

void pybind_color_map_options(py::module &m) {
    static std::unordered_map<std::string, std::string> colormap_docstrings = {
            {"non_rigid_camera_coordinate",
             "bool: (Default ``False``) Set to ``True`` to enable non-rigid "
             "optimization (optimizing camera extrinsic params and image "
             "wrapping field for color assignment), set to ``False`` to only "
             "enable rigid optimization (optimize camera extrinsic params)."},
            {"number_of_vertical_anchors",
             "int: (Default ``16``) Number of vertical anchor points for image "
             "wrapping field. The number of horizontal anchor points is "
             "computed automatically based on the number of vertical anchor "
             "points. This option is only used when non-rigid optimization is "
             "enabled."},
            {"non_rigid_anchor_point_weight",
             "float: (Default ``0.316``) Additional regularization terms added "
             "to non-rigid regularization. A higher value results gives more "
             "conservative updates. If the residual error does not stably "
             "decrease, it is mainly because images are being bended abruptly. "
             "In this case, consider making iteration more conservative by "
             "increasing the value. This option is only used when non-rigid "
             "optimization is enabled."},
            {"maximum_iteration",
             "int: (Default ``300``) Number of iterations for optimization "
             "steps."},
            {"maximum_allowable_depth",
             "float: (Default ``2.5``) Parameter to check the visibility of a "
             "point. Points with depth larger than ``maximum_allowable_depth`` "
             "in a RGB-D will be marked as invisible for the camera producing "
             "that RGB-D image. Select a proper value to include necessary "
             "points while ignoring unwanted points such as the background."},
            {"depth_threshold_for_visibility_check",
             "float: (Default ``0.03``) Parameter for point visibility check. "
             "When the difference of a point's depth value in the RGB-D image "
             "and the point's depth value in the 3D mesh is greater than "
             "``depth_threshold_for_visibility_check``, the point is marked as "
             "invisible to the camera producing the RGB-D image."},
            {"depth_threshold_for_discontinuity_check",
             "float: (Default ``0.1``) Parameter to check the visibility of a "
             "point. It's often desirable to ignore points where there is an "
             "abrupt change in depth value. First the depth gradient image is "
             "computed, points are considered to be invisible if the depth "
             "gradient magnitude is larger than "
             "``depth_threshold_for_discontinuity_check``."},
            {"half_dilation_kernel_size_for_discontinuity_map",
             "int: (Default ``3``) Parameter to check the visibility of a "
             "point. Related to ``depth_threshold_for_discontinuity_check``, "
             "when boundary points are detected, dilation is performed to "
             "ignore points near the object boundary. "
             "``half_dilation_kernel_size_for_discontinuity_map`` specifies "
             "the half-kernel size for the dilation applied on the visibility "
             "mask image."},
            {"image_boundary_margin",
             "int: (Default ``10``) If a projected 3D point onto a 2D image "
             "lies in the image border within ``image_boundary_margin``, the "
             "3D point is considered invisible from the camera producing the "
             "image. This parmeter is not used for visibility check, but used "
             "when computing the final color assignment after color map "
             "optimization."},
            {"invisible_vertex_color_knn",
             "int: (Default ``3``) If a vertex is invisible from all images, "
             "we assign the averaged color of the k nearest visible vertices "
             "to fill the invisible vertex. Set to ``0`` to disable this "
             "feature and all invisible vertices will be black."},
            {"debug_output_dir",
             "If specified, the intermediate results will be stored in in the "
             "debug output dir. Existing files will be overwritten if the "
             "names are the same."}};

    py::class_<pipelines::color_map::RigidOptimizerOption>
            rigid_optimizer_option(m, "RigidOptimizerOption",
                                   "Rigid optimizer option class.");
    rigid_optimizer_option.def(
            py::init([](int maximum_iteration, double maximum_allowable_depth,
                        double depth_threshold_for_visibility_check,
                        double depth_threshold_for_discontinuity_check,
                        int half_dilation_kernel_size_for_discontinuity_map,
                        int image_boundary_margin,
                        int invisible_vertex_color_knn,
                        const std::string &debug_output_dir) {
                auto option = new pipelines::color_map::RigidOptimizerOption;
                option->maximum_iteration_ = maximum_iteration;
                option->maximum_allowable_depth_ = maximum_allowable_depth;
                option->depth_threshold_for_visibility_check_ =
                        depth_threshold_for_visibility_check;
                option->depth_threshold_for_discontinuity_check_ =
                        depth_threshold_for_discontinuity_check;
                option->half_dilation_kernel_size_for_discontinuity_map_ =
                        half_dilation_kernel_size_for_discontinuity_map;
                option->image_boundary_margin_ = image_boundary_margin;
                option->invisible_vertex_color_knn_ =
                        invisible_vertex_color_knn;
                option->debug_output_dir_ = debug_output_dir;
                return option;
            }),
            "maximum_iteration"_a = 0, "maximum_allowable_depth"_a = 2.5,
            "depth_threshold_for_visibility_check"_a = 0.03,
            "depth_threshold_for_discontinuity_check"_a = 0.1,
            "half_dilation_kernel_size_for_discontinuity_map"_a = 3,
            "image_boundary_margin"_a = 10, "invisible_vertex_color_knn"_a = 3,
            "debug_output_dir"_a = "");

    docstring::ClassMethodDocInject(m, "RigidOptimizerOption", "__init__",
                                    colormap_docstrings,
                                    /*skip_init=*/false);

    py::class_<pipelines::color_map::NonRigidOptimizerOption>
            non_rigid_optimizer_option(m, "NonRigidOptimizerOption",
                                       "Non Rigid optimizer option class.");
    non_rigid_optimizer_option.def(
            py::init([](int number_of_vertical_anchors,
                        double non_rigid_anchor_point_weight,
                        int maximum_iteration, double maximum_allowable_depth,
                        double depth_threshold_for_visibility_check,
                        double depth_threshold_for_discontinuity_check,
                        int half_dilation_kernel_size_for_discontinuity_map,
                        int image_boundary_margin,
                        int invisible_vertex_color_knn,
                        const std::string &debug_output_dir) {
                auto option = new pipelines::color_map::NonRigidOptimizerOption;
                option->number_of_vertical_anchors_ =
                        number_of_vertical_anchors;
                option->non_rigid_anchor_point_weight_ =
                        non_rigid_anchor_point_weight;
                option->maximum_iteration_ = maximum_iteration;
                option->maximum_allowable_depth_ = maximum_allowable_depth;
                option->depth_threshold_for_visibility_check_ =
                        depth_threshold_for_visibility_check;
                option->depth_threshold_for_discontinuity_check_ =
                        depth_threshold_for_discontinuity_check;
                option->half_dilation_kernel_size_for_discontinuity_map_ =
                        half_dilation_kernel_size_for_discontinuity_map;
                option->image_boundary_margin_ = image_boundary_margin;
                option->invisible_vertex_color_knn_ =
                        invisible_vertex_color_knn;
                option->debug_output_dir_ = debug_output_dir;
                return option;
            }),
            "number_of_vertical_anchors"_a = 16,
            "non_rigid_anchor_point_weight"_a = 0.316,
            "maximum_iteration"_a = 0, "maximum_allowable_depth"_a = 2.5,
            "depth_threshold_for_visibility_check"_a = 0.03,
            "depth_threshold_for_discontinuity_check"_a = 0.1,
            "half_dilation_kernel_size_for_discontinuity_map"_a = 3,
            "image_boundary_margin"_a = 10, "invisible_vertex_color_knn"_a = 3,
            "debug_output_dir"_a = "");

    docstring::ClassMethodDocInject(m, "NonRigidOptimizerOption", "__init__",
                                    colormap_docstrings,
                                    /*skip_init=*/false);
}

void pybind_color_map_classes(py::module &m) {
    m.def("run_rigid_optimizer", &pipelines::color_map::RunRigidOptimizer,
          "Run rigid optimization.");
    m.def("run_non_rigid_optimizer",
          &pipelines::color_map::RunNonRigidOptimizer,
          "Run non-rigid optimization.");
}

void pybind_color_map(py::module &m) {
    py::module m_submodule =
            m.def_submodule("color_map", "Color map optimization pipeline");
    pybind_color_map_options(m_submodule);
    pybind_color_map_classes(m_submodule);
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
