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

#include "open3d_core.h"
#include "open3d_core_trampoline.h"

#include <Core/Geometry/RGBDImage.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Core/ColorMap/ColorMapOptimization.h>
#include <Core/Camera/PinholeCameraTrajectory.h>

using namespace open3d;

void pybind_colormap_optimization(py::module &m)
{
    py::class_<ColorMapOptmizationOption> color_map_optimization_option(
             m, "ColorMapOptmizationOption");
    py::detail::bind_default_constructor<ColorMapOptmizationOption>(
            color_map_optimization_option);
    color_map_optimization_option
         .def_readwrite("non_rigid_camera_coordinate",
         &ColorMapOptmizationOption::non_rigid_camera_coordinate_)
         .def_readwrite("number_of_vertical_anchors",
         &ColorMapOptmizationOption::number_of_vertical_anchors_)
         .def_readwrite("non_rigid_anchor_point_weight",
         &ColorMapOptmizationOption::non_rigid_anchor_point_weight_)
         .def_readwrite("maximum_iteration",
         &ColorMapOptmizationOption::maximum_iteration_)
         .def_readwrite("maximum_allowable_depth",
         &ColorMapOptmizationOption::maximum_allowable_depth_)
         .def_readwrite("depth_threshold_for_visiblity_check",
         &ColorMapOptmizationOption::depth_threshold_for_visiblity_check_)
         .def_readwrite("depth_threshold_for_discontinuity_check",
         &ColorMapOptmizationOption::depth_threshold_for_discontinuity_check_)
         .def_readwrite("half_dilation_kernel_size_for_discontinuity_map",
         &ColorMapOptmizationOption::half_dilation_kernel_size_for_discontinuity_map_)
        .def("__repr__", [](const ColorMapOptmizationOption &to) {
            return std::string("ColorMapOptmizationOption with") +
                    std::string("\n- non_rigid_camera_coordinate : ") +
                    std::to_string(to.non_rigid_camera_coordinate_) +
                    std::string("\n- number_of_vertical_anchors : ") +
                    std::to_string(to.number_of_vertical_anchors_) +
                    std::string("\n- non_rigid_anchor_point_weight : ") +
                    std::to_string(to.non_rigid_anchor_point_weight_) +
                    std::string("\n- maximum_iteration : ") +
                    std::to_string(to.maximum_iteration_) +
                    std::string("\n- maximum_allowable_depth : ") +
                    std::to_string(to.maximum_allowable_depth_) +
                    std::string("\n- depth_threshold_for_visiblity_check : ") +
                    std::to_string(to.depth_threshold_for_visiblity_check_) +
                    std::string("\n- depth_threshold_for_discontinuity_check : ") +
                    std::to_string(to.depth_threshold_for_discontinuity_check_) +
                    std::string("\n- half_dilation_kernel_size_for_discontinuity_map : ") +
                    std::to_string(to.half_dilation_kernel_size_for_discontinuity_map_);
        });
}

void pybind_colormap_optimization_methods(py::module &m)
{
    m.def("color_map_optimization",
            &ColorMapOptimization,
            "Function for color mapping of reconstructed scenes via optimization",
            "mesh"_a, "imgs_rgbd"_a, "camera"_a,
            "option"_a = ColorMapOptmizationOption());
}
