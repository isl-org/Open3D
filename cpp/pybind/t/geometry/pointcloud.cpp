// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/t/geometry/PointCloud.h"

#include <string>
#include <unordered_map>

#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_pointcloud(py::module& m) {
    py::class_<PointCloud, PyGeometry<PointCloud>, std::unique_ptr<PointCloud>,
               Geometry>
            pointcloud(m, "PointCloud",
                       "A pointcloud contains a set of 3D points.");

    // Constructors.
    pointcloud.def(py::init<const core::Device&>(), "device"_a)
            .def(py::init<const core::Tensor&>(), "points"_a)
            .def(py::init<const std::unordered_map<std::string,
                                                   core::Tensor>&>(),
                 "map_keys_to_tensors"_a);

    // Point's attributes: points, colors, normals, etc.
    // def_property_readonly is sufficient, since the returned TensorMap can
    // be editable in Python. We don't want the TensorMap to be replaced
    // by another TensorMap in Python.
    pointcloud.def_property_readonly(
            "point",
            py::overload_cast<>(&PointCloud::GetPointAttr, py::const_));

    // Device transfers.
    pointcloud.def("to", &PointCloud::To,
                   "Transfer the point cloud to a specified device.",
                   "device"_a, "copy"_a = false);
    pointcloud.def("clone", &PointCloud::Clone,
                   "Returns copy of the point cloud on the same device.");
    pointcloud.def("cpu", &PointCloud::CPU,
                   "Transfer the point cloud to CPU. If the point cloud is "
                   "already on CPU, no copy will be performed.");
    pointcloud.def(
            "cuda", &PointCloud::CUDA,
            "Transfer the point cloud to a CUDA device. If the point cloud is "
            "already on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);

    // Pointcloud specific functions.
    pointcloud.def("get_min_bound", &PointCloud::GetMinBound,
                   "Returns the min bound for point coordinates.");
    pointcloud.def("get_max_bound", &PointCloud::GetMaxBound,
                   "Returns the max bound for point coordinates.");
    pointcloud.def("get_center", &PointCloud::GetCenter,
                   "Returns the center for point coordinates.");
    pointcloud.def("transform", &PointCloud::Transform, "transformation"_a,
                   "Transforms the points and normals (if exist).");
    pointcloud.def("translate", &PointCloud::Translate, "translation"_a,
                   "relative"_a = true, "Translates points.");
    pointcloud.def("scale", &PointCloud::Scale, "scale"_a, "center"_a,
                   "Scale points.");
    pointcloud.def("rotate", &PointCloud::Rotate, "R"_a, "center"_a,
                   "Rotate points and normals (if exist).");
    pointcloud.def_static(
            "create_from_depth_image", &PointCloud::CreateFromDepthImage,
            "depth"_a, "intrinsics"_a,
            "extrinsics"_a = core::Tensor::Eye(4, core::Dtype::Float32,
                                               core::Device("CPU:0")),
            "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f, "stride"_a = 1);
    pointcloud.def_static(
            "from_legacy_pointcloud", &PointCloud::FromLegacyPointCloud,
            "pcd_legacy"_a, "dtype"_a = core::Dtype::Float32,
            "device"_a = core::Device("CPU:0"),
            "Create a PointCloud from a legacy Open3D PointCloud.");
    pointcloud.def("to_legacy_pointcloud", &PointCloud::ToLegacyPointCloud,
                   "Convert to a legacy Open3D PointCloud.");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
