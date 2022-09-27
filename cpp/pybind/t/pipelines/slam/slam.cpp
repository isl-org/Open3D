// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/pipelines/slam/Frame.h"
#include "open3d/t/pipelines/slam/Model.h"
#include "pybind/docstring.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slam {

static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"voxel_size", "The voxel size of the volume in meters."},
                {"block_resolution",
                 "Resolution of local dense voxel blocks. By default 16 "
                 "is used to create 16^3 voxel blocks."},
                {"block_count",
                 "Number of estimate blocks per scene with the block "
                 "resolution set to 16 and the 6mm voxel resolution. "
                 "Typically 20000 for small scenes (a desk), 40000 for medium "
                 "scenes (a bedroom), 80000 for large scenes (an "
                 "apartment)."},
                {"transformation", "A 4x4 3D transformation matrix."},
                {"device", "The CPU or CUDA device used for the object."},
                {"depth_max",
                 "The max clipping depth to filter noisy observations too "
                 "far."},
                {"depth_min", "The min clipping depth."},
                {"depth_scale",
                 "The scale factor to convert raw depth into meters."},
                {"input_frame",
                 "The frame that contains raw depth and optionally images with "
                 "the same size from the input."},
                {"model_frame",
                 "The frame that contains ray casted depth and optionally "
                 "color from the volumetric model."},
                {"estimated_number",
                 "Estimated number of surface points. Use -1 if no estimation "
                 "is available."},
                {"weight_threshold",
                 "Weight threshold to filter outlier voxel blocks."},
                {"height", "Height of an image frame."},
                {"width", "Width of an image frame."},
                {"intrinsics", "Intrinsic matrix stored in a 3x3 Tensor."},
                {"trunc_voxel_multiplier",
                 "Truncation distance multiplier in voxel size for signed "
                 "distance. For instance, "
                 "--trunc_voxel_multiplier=8 with --voxel_size=0.006(m) "
                 "creates a truncation distance of 0.048(m)."}};

void pybind_slam_model(py::module &m) {
    py::class_<Model> model(m, "Model", "Volumetric model for Dense SLAM.");
    py::detail::bind_copy_functions<Model>(model);

    model.def(py::init<>());
    model.def(py::init<float, int, int, core::Tensor, core::Device>(),
              "Constructor of a VoxelBlockGrid", "voxel_size"_a,
              "block_resolution"_a = 16, " block_count"_a = 10000,
              "transformation"_a = core::Tensor::Eye(4, core::Float64,
                                                     core::Device("CPU:0")),
              "device"_a = core::Device("CUDA:0"));
    docstring::ClassMethodDocInject(m, "Model", "__init__",
                                    map_shared_argument_docstrings);

    model.def("get_current_frame_pose", &Model::GetCurrentFramePose);
    model.def("update_frame_pose", &Model::UpdateFramePose);

    model.def("synthesize_model_frame", &Model::SynthesizeModelFrame,
              py::call_guard<py::gil_scoped_release>(),
              "Synthesize frame from the volumetric model using ray casting.",
              "model_frame"_a, "depth_scale"_a = 1000.0, "depth_min"_a = 0.1,
              "depth_max"_a = 3.0, "trunc_voxel_multiplier"_a = 8.0,
              "enable_color"_a = false, "weight_threshold"_a = -1.0);
    docstring::ClassMethodDocInject(m, "Model", "synthesize_model_frame",
                                    map_shared_argument_docstrings);

    model.def(
            "track_frame_to_model", &Model::TrackFrameToModel,
            py::call_guard<py::gil_scoped_release>(),
            "Track input frame against raycasted frame from model.",
            "input_frame"_a, "model_frame"_a, "depth_scale"_a = 1000.0,
            "depth_max"_a = 3.0, "depth_diff"_a = 0.07,
            "method"_a = odometry::Method::PointToPlane,
            "criteria"_a = (std::vector<odometry::OdometryConvergenceCriteria>){
                    6, 3, 1});
    docstring::ClassMethodDocInject(m, "Model", "track_frame_to_model",
                                    map_shared_argument_docstrings);

    model.def("integrate", &Model::Integrate,
              py::call_guard<py::gil_scoped_release>(),
              "Integrate an input frame to a volume.", "input_frame"_a,
              "depth_scale"_a = 1000.0, "depth_max"_a = 3.0,
              "trunc_voxel_multiplier"_a = 8.0);
    docstring::ClassMethodDocInject(m, "Model", "integrate",
                                    map_shared_argument_docstrings);

    model.def("extract_pointcloud", &Model::ExtractPointCloud,
              py::call_guard<py::gil_scoped_release>(),
              "Extract point cloud from the volumetric model.",
              "weight_threshold"_a = 3.0, "estimated_number"_a = -1);
    docstring::ClassMethodDocInject(m, "Model", "extract_pointcloud",
                                    map_shared_argument_docstrings);

    model.def("extract_trianglemesh", &Model::ExtractTriangleMesh,
              py::call_guard<py::gil_scoped_release>(),
              "Extract triangle mesh from the volumetric model.",
              "weight_threshold"_a = 3.0, "estimated_number"_a = -1);
    docstring::ClassMethodDocInject(m, "Model", "extract_trianglemesh",
                                    map_shared_argument_docstrings);

    model.def(
            "get_hashmap", &Model::GetHashMap,
            "Get the underlying hash map from 3D coordinates to voxel blocks.");
    model.def_readwrite("voxel_grid", &Model::voxel_grid_,
                        "Get the maintained VoxelBlockGrid.");
    model.def_readwrite("frustum_block_coords", &Model::frustum_block_coords_,
                        "Active block coordinates from prior integration");
    model.def_readwrite("transformation_frame_to_world",
                        &Model::T_frame_to_world_,
                        "Get the 4x4 transformation matrix from the current "
                        "frame to the world frame.");
    model.def_readwrite("frame_id", &Model::frame_id_,
                        "Get the current frame index in a sequence.");
}

void pybind_slam_frame(py::module &m) {
    py::class_<Frame> frame(m, "Frame",
                            "A frame container that stores a map from keys "
                            "(color, depth) to tensor images.");
    py::detail::bind_copy_functions<Frame>(frame);

    frame.def(py::init<int, int, core::Tensor, core::Device>(), "height"_a,
              "width"_a, "intrinsics"_a, "device"_a);
    docstring::ClassMethodDocInject(m, "Frame", "__init__",
                                    map_shared_argument_docstrings);

    frame.def("height", &Frame::GetHeight);
    frame.def("width", &Frame::GetWidth);

    frame.def("set_data", &Frame::SetData,
              "Set a 2D tensor to a image to the given key in the map.");
    frame.def("get_data", &Frame::GetData,
              "Get a 2D tensor from a image from the given key in the map.");
    frame.def("set_data_from_image", &Frame::SetDataFromImage,
              "Set a 2D image to the given key in the map.");
    frame.def("get_data_as_image", &Frame::GetDataAsImage,
              "Get a 2D image from from the given key in the map.");
}

void pybind_slam(py::module &m) {
    py::module m_submodule =
            m.def_submodule("slam", "Tensor DenseSLAM pipeline.");
    pybind_slam_model(m_submodule);
    pybind_slam_frame(m_submodule);
}

}  // namespace slam
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
