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

void pybind_slam_model(py::module &m) {
    // TODO: doc
    py::class_<Model> model(m, "Model", "DenseSLAM model.");
    py::detail::bind_copy_functions<Model>(model);

    model.def(py::init<>());
    model.def(py::init<float, float, int, int, core::Tensor, core::Device>(),
              "Constructor of a TSDFVoxelGrid", "voxel_size"_a, "sdf_trunc"_a,
              "block_resolution"_a = 16, "block_count"_a = 10000,
              "transformation"_a = core::Tensor::Eye(4, core::Float64,
                                                     core::Device("CPU:0")),
              "device"_a = core::Device("CUDA:0"));

    model.def("get_current_frame_pose", &Model::GetCurrentFramePose);
    model.def("update_frame_pose", &Model::UpdateFramePose);

    model.def("synthesize_model_frame", &Model::SynthesizeModelFrame,
              py::call_guard<py::gil_scoped_release>(),
              "Synthesize frame from the volumetric model using ray casting.",
              "model_frame"_a, "depth_scale"_a = 1000.0, "depth_min"_a = 0.1,
              "depth_max"_a = 3.0, "enable_color"_a = false);

    model.def("track_frame_to_model", &Model::TrackFrameToModel,
              py::call_guard<py::gil_scoped_release>(),
              "Track input frame against raycasted frame from model.",
              "input_frame"_a, "model_frame"_a, "depth_scale"_a = 1000.0,
              "depth_max"_a = 3.0, "depth_diff"_a = 0.07);

    model.def("integrate", &Model::Integrate,
              py::call_guard<py::gil_scoped_release>(),
              "Integrate an input frame to a volume.", "input_frame"_a,
              "depth_scale"_a = 1000.0, "depth_max"_a = 3.0);

    model.def("extract_pointcloud", &Model::ExtractPointCloud,
              py::call_guard<py::gil_scoped_release>(),
              "Extract point cloud from the volumetric model.",
              "estimated_number"_a = -1, "weight_threshold"_a = 3.0);

    model.def("extract_trianglemesh", &Model::ExtractTriangleMesh,
              py::call_guard<py::gil_scoped_release>(),
              "Extract triangle mesh from the volumetric model.",
              "estimated_number"_a = -1, "weight_threshold"_a = 3.0);

    model.def("get_hashmap", &Model::GetHashmap);

    model.def_readwrite("voxel_grid", &Model::voxel_grid_);
    model.def_readwrite("T_frame_to_world", &Model::T_frame_to_world_);
    model.def_readwrite("frame_id", &Model::frame_id_);
}

void pybind_slam_frame(py::module &m) {
    // TODO: doc
    py::class_<Frame> frame(m, "Frame", "DenseSLAM frame.");
    py::detail::bind_copy_functions<Frame>(frame);

    frame.def(py::init<int, int, core::Tensor, core::Device>(), "height"_a,
              "width"_a, "intrinsics"_a, "device"_a);

    frame.def("height", &Frame::GetHeight);
    frame.def("width", &Frame::GetWidth);
    frame.def("set_data", &Frame::SetData);
    frame.def("get_data", &Frame::GetData);

    frame.def("set_data_from_image", &Frame::SetDataFromImage);
    frame.def("get_data_as_image", &Frame::GetDataAsImage);
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
