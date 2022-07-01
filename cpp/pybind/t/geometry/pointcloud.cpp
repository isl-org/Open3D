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

#include "open3d/t/geometry/PointCloud.h"

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

// Image functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"rgbd_image",
                 "The input RGBD image should have a uint16_t depth image and  "
                 "RGB image with any DType and the same size."},
                {"depth", "The input depth image should be a uint16_t image."},
                {"intrinsics", "Intrinsic parameters of the camera."},
                {"extrinsics", "Extrinsic parameters of the camera."},
                {"depth_scale", "The depth is scaled by 1 / depth_scale."},
                {"depth_max", "Truncated at depth_max distance."},
                {"stride",
                 "Sampling factor to support coarse point cloud extraction. "
                 "Unless normals are requested, there is no low pass "
                 "filtering, so aliasing is possible for stride>1."},
                {"with_normals",
                 "Also compute normals for the point cloud. If True, the point "
                 "cloud will only contain points with valid normals. If "
                 "normals are requested, the depth map is first filtered to "
                 "ensure smooth normals."},
                {"max_nn",
                 "Neighbor search max neighbors parameter [default = 30]."},
                {"radius",
                 "neighbors search radius parameter to use HybridSearch. "
                 "[Recommended ~1.4x voxel size]."}};

void pybind_pointcloud(py::module& m) {
    py::class_<PointCloud, PyGeometry<PointCloud>, std::shared_ptr<PointCloud>,
               Geometry, DrawableGeometry>
            pointcloud(m, "PointCloud",
                       R"(
A point cloud contains a list of 3D points. The point cloud class stores the
attribute data in key-value maps, where the key is a string representing the
attribute name and the value is a Tensor containing the attribute data.

The attributes of the point cloud have different levels::

    import open3d as o3d

    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32

    # Create an empty point cloud
    # Use pcd.point to access the points' attributes
    pcd = o3d.t.geometry.PointCloud(device)

    # Default attribute: "positions".
    # This attribute is created by default and is required by all point clouds.
    # The shape must be (N, 3). The device of "positions" determines the device
    # of the point cloud.
    pcd.point["positions"] = o3d.core.Tensor([[0, 0, 0],
                                              [1, 1, 1],
                                              [2, 2, 2]], dtype, device)

    # Common attributes: "normals", "colors".
    # Common attributes are used in built-in point cloud operations. The
    # spellings must be correct. For example, if "normal" is used instead of
    # "normals", some internal operations that expects "normals" will not work.
    # "normals" and "colors" must have shape (N, 3) and must be on the same
    # device as the point cloud.
    pcd.point["normals"] = o3d.core.Tensor([[0, 0, 1],
                                            [0, 1, 0],
                                            [1, 0, 0]], dtype, device)
    pcd.point["colors"] = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                            [0.1, 0.1, 0.1],
                                            [0.2, 0.2, 0.2]], dtype, device)

    # User-defined attributes.
    # You can also attach custom attributes. The value tensor must be on the
    # same device as the point cloud. The are no restrictions on the shape and
    # dtype, e.g.,
    pcd.point["intensities"] = o3d.core.Tensor([0.3, 0.1, 0.4], dtype, device)
    pcd.point["labels"] = o3d.core.Tensor([3, 1, 4], o3d.core.int32, device)
)");

    // Constructors.
    pointcloud
            .def(py::init<const core::Device&>(),
                 "Construct an empty pointcloud on the provided ``device`` "
                 "(default: 'CPU:0').",
                 "device"_a = core::Device("CPU:0"))
            .def(py::init<const core::Tensor&>(), "positions"_a)
            .def(py::init<const std::unordered_map<std::string,
                                                   core::Tensor>&>(),
                 "map_keys_to_tensors"_a)
            .def("__repr__", &PointCloud::ToString);

    // def_property_readonly is sufficient, since the returned TensorMap can
    // be editable in Python. We don't want the TensorMap to be replaced
    // by another TensorMap in Python.
    pointcloud.def_property_readonly(
            "point", py::overload_cast<>(&PointCloud::GetPointAttr, py::const_),
            "Point's attributes: positions, colors, normals, etc.");

    // Device transfers.
    pointcloud.def("to", &PointCloud::To,
                   "Transfer the point cloud to a specified device.",
                   "device"_a, "copy"_a = false);
    pointcloud.def("clone", &PointCloud::Clone,
                   "Returns a copy of the point cloud on the same device.");

    pointcloud.def(
            "cpu",
            [](const PointCloud& pointcloud) {
                return pointcloud.To(core::Device("CPU:0"));
            },
            "Transfer the point cloud to CPU. If the point cloud is "
            "already on CPU, no copy will be performed.");
    pointcloud.def(
            "cuda",
            [](const PointCloud& pointcloud, int device_id) {
                return pointcloud.To(core::Device("CUDA", device_id));
            },
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

    pointcloud.def("append",
                   [](const PointCloud& self, const PointCloud& other) {
                       return self.Append(other);
                   });
    pointcloud.def("__add__",
                   [](const PointCloud& self, const PointCloud& other) {
                       return self.Append(other);
                   });

    pointcloud.def("transform", &PointCloud::Transform, "transformation"_a,
                   "Transforms the points and normals (if exist).");
    pointcloud.def("translate", &PointCloud::Translate, "translation"_a,
                   "relative"_a = true, "Translates points.");
    pointcloud.def("scale", &PointCloud::Scale, "scale"_a, "center"_a,
                   "Scale points.");
    pointcloud.def("rotate", &PointCloud::Rotate, "R"_a, "center"_a,
                   "Rotate points and normals (if exist).");

    pointcloud.def("select_by_mask", &PointCloud::SelectByMask,
                   "boolean_mask"_a, "invert"_a = false,
                   "Select points from input pointcloud, based on boolean mask "
                   "indices into output point cloud.");
    pointcloud.def("select_by_index", &PointCloud::SelectByIndex, "indices"_a,
                   "invert"_a = false, "remove_duplicates"_a = false,
                   "Select points from input pointcloud, based on indices into "
                   "output point cloud.");
    pointcloud.def(
            "voxel_down_sample",
            [](const PointCloud& pointcloud, const double voxel_size) {
                return pointcloud.VoxelDownSample(
                        voxel_size, core::HashBackendType::Default);
            },
            "Downsamples a point cloud with a specified voxel size.",
            "voxel_size"_a);
    pointcloud.def("uniform_down_sample", &PointCloud::UniformDownSample,
                   "Downsamples a point cloud by selecting every kth index "
                   "point and its attributes.",
                   "every_k_points"_a);
    pointcloud.def("random_down_sample", &PointCloud::RandomDownSample,
                   "Downsample a pointcloud by selecting random index point "
                   "and its attributes.",
                   "sampling_ratio"_a);
    pointcloud.def("remove_radius_outliers", &PointCloud::RemoveRadiusOutliers,
                   "nb_points"_a, "search_radius"_a,
                   "Remove points that have less than nb_points neighbors in a "
                   "sphere of a given search radius.");

    pointcloud.def("estimate_normals", &PointCloud::EstimateNormals,
                   py::call_guard<py::gil_scoped_release>(),
                   py::arg("max_nn") = 30, py::arg("radius") = py::none(),
                   "Function to estimate point normals. If the pointcloud "
                   "normals exists, the estimated normals are oriented "
                   "with respect to the same. It uses KNN search if only "
                   "max_nn parameter is provided, and HybridSearch if radius "
                   "parameter is also provided.");
    pointcloud.def("estimate_color_gradients",
                   &PointCloud::EstimateColorGradients,
                   py::call_guard<py::gil_scoped_release>(),
                   py::arg("max_nn") = 30, py::arg("radius") = py::none(),
                   "Function to estimate point color gradients. If radius is "
                   "provided, then HybridSearch is used, otherwise KNN-Search "
                   "is used.");

    // creation (static)
    pointcloud.def_static(
            "create_from_depth_image", &PointCloud::CreateFromDepthImage,
            py::call_guard<py::gil_scoped_release>(), "depth"_a, "intrinsics"_a,
            "extrinsics"_a =
                    core::Tensor::Eye(4, core::Float32, core::Device("CPU:0")),
            "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f, "stride"_a = 1,
            "with_normals"_a = false,
            "Factory function to create a pointcloud (with only 'points') from "
            "a depth image and a camera model.\n\n Given depth value d at (u, "
            "v) image coordinate, the corresponding 3d point is:\n z = d / "
            "depth_scale\n\n x = (u - cx) * z / fx\n\n y = (v - cy) * z / fy");
    pointcloud.def_static(
            "create_from_rgbd_image", &PointCloud::CreateFromRGBDImage,
            py::call_guard<py::gil_scoped_release>(), "rgbd_image"_a,
            "intrinsics"_a,
            "extrinsics"_a =
                    core::Tensor::Eye(4, core::Float32, core::Device("CPU:0")),
            "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f, "stride"_a = 1,
            "with_normals"_a = false,
            "Factory function to create a pointcloud (with properties "
            "{'points', 'colors'}) from an RGBD image and a camera model.\n\n"
            "Given depth value d at (u, v) image coordinate, the corresponding "
            "3d point is:\n\n z = d / depth_scale\n\n x = (u - cx) * z / "
            "fx\n\n y "
            "= (v - cy) * z / fy");
    pointcloud.def_static(
            "from_legacy", &PointCloud::FromLegacy, "pcd_legacy"_a,
            "dtype"_a = core::Float32, "device"_a = core::Device("CPU:0"),
            "Create a PointCloud from a legacy Open3D PointCloud.");

    // processing
    pointcloud.def("project_to_depth_image", &PointCloud::ProjectToDepthImage,
                   "width"_a, "height"_a, "intrinsics"_a,
                   "extrinsics"_a = core::Tensor::Eye(4, core::Float32,
                                                      core::Device("CPU:0")),
                   "depth_scale"_a = 1000.0, "depth_max"_a = 3.0,
                   "Project a point cloud to a depth image.");
    pointcloud.def("project_to_rgbd_image", &PointCloud::ProjectToRGBDImage,
                   "width"_a, "height"_a, "intrinsics"_a,
                   "extrinsics"_a = core::Tensor::Eye(4, core::Float32,
                                                      core::Device("CPU:0")),
                   "depth_scale"_a = 1000.0, "depth_max"_a = 3.0,
                   "Project a colored point cloud to a RGBD image.");
    pointcloud.def(
            "cluster_dbscan", &PointCloud::ClusterDBSCAN,
            "Cluster PointCloud using the DBSCAN algorithm  Ester et al., "
            "'A Density-Based Algorithm for Discovering Clusters in Large "
            "Spatial Databases with Noise', 1996. Returns a list of point "
            "labels, -1 indicates noise according to the algorithm.",
            "eps"_a, "min_points"_a, "print_progress"_a = false);
    pointcloud.def(
            "compute_convex_hull", &PointCloud::ComputeConvexHull,
            "joggle_inputs"_a = false,
            R"doc(Compute the convex hull of a triangle mesh using qhull. This runs on the CPU.

Args:
    joggle_inputs (default False). Handle precision problems by
    randomly perturbing the input data. Set to True if perturbing the input
    iis acceptable but you need convex simplicial output. If False,
    neighboring facets may be merged in case of precision problems. See
    `QHull docs <http://www.qhull.org/html/qh-impre.htm#joggle>`__ for more
    details.

Return:
    TriangleMesh representing the convexh hull. This contains an
    extra vertex property "point_indices" that contains the index of the
    corresponding vertex in the original mesh.

Example:
    We will load the Eagle dataset, compute and display it's convex hull::

        eagle = o3d.data.EaglePointCloud()
        pcd = o3d.t.io.read_point_cloud(eagle.path)
        hull = pcd.compute_convex_hull()
        o3d.visualization.draw([{'name': 'eagle', 'geometry': pcd}, {'name': 'convex hull', 'geometry': hull}])
    )doc");

    // conversion
    pointcloud.def("to_legacy", &PointCloud::ToLegacy,
                   "Convert to a legacy Open3D PointCloud.");

    docstring::ClassMethodDocInject(m, "PointCloud", "estimate_normals",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "PointCloud", "create_from_depth_image",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(m, "PointCloud", "create_from_rgbd_image",
                                    map_shared_argument_docstrings);
    docstring::ClassMethodDocInject(
            m, "PointCloud", "select_by_mask",
            {{"boolean_mask",
              "Boolean indexing tensor of shape {n,} containing true value for "
              "the indices that is to be selected.."},
             {"invert", "Set to `True` to invert the selection of indices."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "select_by_index",
            {{"indices",
              "Int64 indexing tensor of shape {n,} containing index value that "
              "is to be selected."},
             {"invert",
              "Set to `True` to invert the selection of indices, and also "
              "ignore the duplicated indices."},
             {"remove_duplicates",
              "Set to `True` to remove the duplicated indices."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "voxel_down_sample",
            {{"voxel_size", "Voxel size. A positive number."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "uniform_down_sample",
            {{"every_k_points",
              "Sample rate, the selected point indices are [0, k, 2k, …]."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "random_down_sample",
            {{"sampling_ratio",
              "Sampling ratio, the ratio of sample to total number of points "
              "in the pointcloud."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_radius_outliers",
            {{"nb_points",
              "Number of neighbor points required within the radius."},
             {"search_radius", "Radius of the sphere."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "cluster_dbscan",
            {{"eps",
              "Density parameter that is used to find neighbouring points."},
             {"min_points", "Minimum number of points to form a cluster."},
             {"print_progress",
              "If true the progress is visualized in the console."}});
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
