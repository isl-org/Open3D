// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/PointCloud.h"

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/LineSet.h"
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

void pybind_pointcloud_declarations(py::module& m) {
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
    pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
                                           [1, 1, 1],
                                           [2, 2, 2]], dtype, device)

    # Common attributes: "normals", "colors".
    # Common attributes are used in built-in point cloud operations. The
    # spellings must be correct. For example, if "normal" is used instead of
    # "normals", some internal operations that expects "normals" will not work.
    # "normals" and "colors" must have shape (N, 3) and must be on the same
    # device as the point cloud.
    pcd.point.normals = o3d.core.Tensor([[0, 0, 1],
                                         [0, 1, 0],
                                         [1, 0, 0]], dtype, device)
    pcd.point.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                        [0.1, 0.1, 0.1],
                                        [0.2, 0.2, 0.2]], dtype, device)

    # User-defined attributes.
    # You can also attach custom attributes. The value tensor must be on the
    # same device as the point cloud. The are no restrictions on the shape and
    # dtype, e.g.,
    pcd.point.intensities = o3d.core.Tensor([0.3, 0.1, 0.4], dtype, device)
    pcd.point.labels = o3d.core.Tensor([3, 1, 4], o3d.core.int32, device)
)");
}
void pybind_pointcloud_definitions(py::module& m) {
    auto pointcloud =
            static_cast<py::class_<PointCloud, PyGeometry<PointCloud>,
                                   std::shared_ptr<PointCloud>, Geometry,
                                   DrawableGeometry>>(m.attr("PointCloud"));
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

    py::detail::bind_copy_functions<PointCloud>(pointcloud);

    // Pickle support.
    pointcloud.def(py::pickle(
            [](const PointCloud& pcd) {
                // __getstate__
                // Convert point attributes to tensor map to CPU.
                auto map_keys_to_tensors = pcd.GetPointAttr();

                return py::make_tuple(pcd.GetDevice(), pcd.GetPointAttr());
            },
            [](py::tuple t) {
                // __setstate__
                if (t.size() != 2) {
                    utility::LogError(
                            "Cannot unpickle PointCloud! Expecting a tuple of "
                            "size 2.");
                }

                const core::Device device = t[0].cast<core::Device>();
                PointCloud pcd(device);
                if (!device.IsAvailable()) {
                    utility::LogWarning(
                            "Device ({}) is not available. PointCloud will be "
                            "created on CPU.",
                            device.ToString());
                    pcd.To(core::Device("CPU:0"));
                }

                const TensorMap map_keys_to_tensors = t[1].cast<TensorMap>();
                for (auto& kv : map_keys_to_tensors) {
                    pcd.SetPointAttr(kv.first, kv.second);
                }

                return pcd;
            }));

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
            [](const PointCloud& pointcloud, const double voxel_size,
               const std::string& reduction) {
                return pointcloud.VoxelDownSample(voxel_size, reduction);
            },
            "Downsamples a point cloud with a specified voxel size and a "
            "reduction type.",
            "voxel_size"_a, "reduction"_a = "mean",
            R"doc(Downsamples a point cloud with a specified voxel size.

Args:
    voxel_size (float): The size of the voxel used to downsample the point cloud.

    reduction (str): The approach to pool point properties in a voxel. Can only be "mean" at current.

Return:
    A downsampled point cloud with point properties reduced in each voxel.

Example:

    We will load the Eagle dataset, downsample it, and show the result::

        eagle = o3d.data.EaglePointCloud()
        pcd = o3d.t.io.read_point_cloud(eagle.path)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
        o3d.visualization.draw([{'name': 'pcd', 'geometry': pcd}, {'name': 'pcd_down', 'geometry': pcd_down}])
    )doc");
    pointcloud.def("uniform_down_sample", &PointCloud::UniformDownSample,
                   "Downsamples a point cloud by selecting every kth index "
                   "point and its attributes.",
                   "every_k_points"_a);
    pointcloud.def("random_down_sample", &PointCloud::RandomDownSample,
                   "Downsample a pointcloud by selecting random index point "
                   "and its attributes.",
                   "sampling_ratio"_a);
    pointcloud.def("farthest_point_down_sample",
                   &PointCloud::FarthestPointDownSample,
                   "Downsample a pointcloud into output pointcloud with a set "
                   "of points has farthest distance.The sampling is performed "
                   "by selecting the farthest point from previous selected "
                   "points iteratively",
                   "num_samples"_a,
                   "Index to start downsampling from. Valid index is a "
                   "non-negative number less than number of points in the "
                   "input pointcloud.",
                   "start_index"_a = 0);
    pointcloud.def("remove_radius_outliers", &PointCloud::RemoveRadiusOutliers,
                   "nb_points"_a, "search_radius"_a,
                   R"(Remove points that have less than nb_points neighbors in a
sphere of a given search radius.

Args:
    nb_points: Number of neighbor points required within the radius.
    search_radius: Radius of the sphere.

Return:
    Tuple of filtered point cloud and boolean mask tensor for selected values
    w.r.t. input point cloud.)");
    pointcloud.def(
            "remove_statistical_outliers",
            &PointCloud::RemoveStatisticalOutliers, "nb_neighbors"_a,
            "std_ratio"_a,
            R"(Remove points that are further away from their \p nb_neighbor
neighbors in average. This function is not recommended to use on GPU.

Args:
    nb_neighbors: Number of neighbors around the target point.
    std_ratio: Standard deviation ratio.

Return:
    Tuple of filtered point cloud and boolean mask tensor for selected values
    w.r.t. input point cloud.)");
    pointcloud.def("remove_duplicated_points",
                   &PointCloud::RemoveDuplicatedPoints,
                   "Remove duplicated points and there associated attributes.");
    pointcloud.def(
            "remove_non_finite_points", &PointCloud::RemoveNonFinitePoints,
            "remove_nan"_a = true, "remove_infinite"_a = true,
            R"(Remove all points from the point cloud that have a nan entry, or
infinite value. It also removes the corresponding attributes.

Args:
    remove_nan: Remove NaN values from the PointCloud.
    remove_infinite: Remove infinite values from the PointCloud.

Return:
    Tuple of filtered point cloud and boolean mask tensor for selected values
    w.r.t. input point cloud.)");
    pointcloud.def("paint_uniform_color", &PointCloud::PaintUniformColor,
                   "color"_a, "Assigns uniform color to the point cloud.");

    pointcloud.def("normalize_normals", &PointCloud::NormalizeNormals,
                   "Normalize point normals to length 1.");
    pointcloud.def(
            "estimate_normals", &PointCloud::EstimateNormals,
            py::call_guard<py::gil_scoped_release>(), py::arg("max_nn") = 30,
            py::arg("radius") = py::none(),
            "Function to estimate point normals. If the point cloud normals "
            "exist, the estimated normals are oriented with respect to the "
            "same. It uses KNN search (Not recommended to use on GPU) if only "
            "max_nn parameter is provided, Radius search (Not recommended to "
            "use on GPU) if only radius is provided and Hybrid Search "
            "(Recommended) if radius parameter is also provided.");
    pointcloud.def("orient_normals_to_align_with_direction",
                   &PointCloud::OrientNormalsToAlignWithDirection,
                   "Function to orient the normals of a point cloud.",
                   "orientation_reference"_a = core::Tensor::Init<float>(
                           {0, 0, 1}, core::Device("CPU:0")));
    pointcloud.def("orient_normals_towards_camera_location",
                   &PointCloud::OrientNormalsTowardsCameraLocation,
                   "Function to orient the normals of a point cloud.",
                   "camera_location"_a = core::Tensor::Zeros(
                           {3}, core::Float32, core::Device("CPU:0")));
    pointcloud.def(
            "orient_normals_consistent_tangent_plane",
            &PointCloud::OrientNormalsConsistentTangentPlane, "k"_a,
            "lambda"_a = 0.0, "cos_alpha_tol"_a = 1.0,
            R"(Function to consistently orient the normals of a point cloud based on tangent planes.

The algorithm is described in Hoppe et al., "Surface Reconstruction from Unorganized Points", 1992.
Additional information about the choice of lambda and cos_alpha_tol for complex
point clouds can be found in Piazza, Valentini, Varetti, "Mesh Reconstruction from Point Cloud", 2023
(https://eugeniovaretti.github.io/meshreco/Piazza_Valentini_Varetti_MeshReconstructionFromPointCloud_2023.pdf).

Args:
    k (int): Number of neighbors to use for tangent plane estimation.
    lambda (float): A non-negative real parameter that influences the distance
        metric used to identify the true neighbors of a point in complex
        geometries. It penalizes the distance between a point and the tangent
        plane defined by the reference point and its normal vector, helping to
        mitigate misclassification issues encountered with traditional
        Euclidean distance metrics.
    cos_alpha_tol (float): Cosine threshold angle used to determine the
        inclusion boundary of neighbors based on the direction of the normal
        vector.

Example:
    We use Bunny point cloud to compute its normals and orient them consistently.
    The initial reconstruction adheres to Hoppe's algorithm (raw), whereas the
    second reconstruction utilises the lambda and cos_alpha_tol parameters.
    Due to the high density of the Bunny point cloud available in Open3D a larger
    value of the parameter k is employed to test the algorithm.  Usually you do
    not have at disposal such a refined point clouds, thus you cannot find a
    proper choice of k: refer to
    https://eugeniovaretti.github.io/meshreco for these cases.::

        import open3d as o3d
        import numpy as np
        # Load point cloud
        data = o3d.data.BunnyMesh()

        # Case 1, Hoppe (raw):
        pcd = o3d.io.read_point_cloud(data.path)

        # Compute normals and orient them consistently, using k=100 neighbours
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

        # Create mesh from point cloud using Poisson Algorithm
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        poisson_mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        poisson_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([poisson_mesh])

        # Case 2, reconstruction using lambda and cos_alpha_tol parameters:
        pcd_robust = o3d.io.read_point_cloud(data.path)

        # Compute normals and orient them consistently, using k=100 neighbours
        pcd_robust.estimate_normals()
        pcd_robust.orient_normals_consistent_tangent_plane(100, 10, 0.5)

        # Create mesh from point cloud using Poisson Algorithm
        poisson_mesh_robust = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_robust, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        poisson_mesh_robust.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        poisson_mesh_robust.compute_vertex_normals()

        o3d.visualization.draw_geometries([poisson_mesh_robust]) )");
    pointcloud.def(
            "estimate_color_gradients", &PointCloud::EstimateColorGradients,
            py::call_guard<py::gil_scoped_release>(), py::arg("max_nn") = 30,
            py::arg("radius") = py::none(),
            "Function to estimate point color gradients. It uses KNN search "
            "(Not recommended to use on GPU) if only max_nn parameter is "
            "provided, Radius search (Not recommended to use on GPU) if only "
            "radius is provided and Hybrid Search (Recommended) if radius "
            "parameter is also provided.");

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
            "v) image coordinate, the corresponding 3d point is:\n\n z = d / "
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
                   py::arg_v("extrinsics",
                             core::Tensor::Eye(4, core::Float32,
                                               core::Device("CPU:0")),
                             "open3d.core.Tensor.eye(4)"),
                   "depth_scale"_a = 1000.0, "depth_max"_a = 3.0,
                   "Project a point cloud to a depth image.");
    pointcloud.def("project_to_rgbd_image", &PointCloud::ProjectToRGBDImage,
                   "width"_a, "height"_a, "intrinsics"_a,
                   py::arg_v("extrinsics",
                             core::Tensor::Eye(4, core::Float32,
                                               core::Device("CPU:0")),
                             "open3d.core.Tensor.eye(4)"),
                   "depth_scale"_a = 1000.0, "depth_max"_a = 3.0,
                   "Project a colored point cloud to a RGBD image.");
    pointcloud.def(
            "hidden_point_removal", &PointCloud::HiddenPointRemoval,
            "camera_location"_a, "radius"_a,
            R"(Removes hidden points from a point cloud and returns a mesh of
the remaining points. Based on Katz et al. 'Direct Visibility of Point Sets',
2007. Additional information about the choice of radius for noisy point clouds
can be found in Mehra et. al. 'Visibility of Noisy Point Cloud Data', 2010.
This is a wrapper for a CPU implementation and a copy of the point cloud data
and resulting visible triangle mesh and indiecs will be made.

Args:
    camera_location: All points not visible from that location will be removed.

    radius: The radius of the spherical projection.

Return:
    Tuple of visible triangle mesh and indices of visible points on the same
    device as the point cloud.

Example:

    We use armadillo mesh to compute the visible points from given camera::

        # Convert mesh to a point cloud and estimate dimensions.
        armadillo_data = o3d.data.ArmadilloMesh()
        pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)

        diameter = np.linalg.norm(
                np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

        # Define parameters used for hidden_point_removal.
        camera = o3d.core.Tensor([0, 0, diameter], o3d.core.float32)
        radius = diameter * 100

        # Get all points that are visible from given view point.
        pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        pcd = pcd.select_by_index(pt_map)
        o3d.visualization.draw([pcd], point_size=5))");
    pointcloud.def(
            "cluster_dbscan", &PointCloud::ClusterDBSCAN, "eps"_a,
            "min_points"_a, "print_progress"_a = false,
            R"(Cluster PointCloud using the DBSCAN algorithm  Ester et al.,'A
Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
with Noise', 1996. This is a wrapper for a CPU implementation and a copy of the
point cloud data and resulting labels will be made.

Args:
    eps: Density parameter that is used to find neighbouring points.

    min_points: Minimum number of points to form a cluster.

print_progress (default False): If 'True' the progress is visualized in the console.

Return:
    A Tensor list of point labels on the same device as the point cloud, -1
    indicates noise according to the algorithm.

Example:

    We use Redwood dataset for demonstration::

        import matplotlib.pyplot as plt

        sample_ply_data = o3d.data.PLYPointCloud()
        pcd = o3d.t.io.read_point_cloud(sample_ply_data.path)
        labels = pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)

        max_label = labels.max().item()
        colors = plt.get_cmap("tab20")(
                labels.numpy() / (max_label if max_label > 0 else 1))
        colors = o3d.core.Tensor(colors[:, :3], o3d.core.float32)
        colors[labels < 0] = 0
        pcd.point.colors = colors
        o3d.visualization.draw([pcd]))");
    pointcloud.def(
            "segment_plane", &PointCloud::SegmentPlane,
            "distance_threshold"_a = 0.01, "ransac_n"_a = 3,
            "num_iterations"_a = 100, "probability"_a = 0.999,
            R"(Segments a plane in the point cloud using the RANSAC algorithm.
This is a wrapper for a CPU implementation and a copy of the point cloud data and
resulting plane model and inlier indiecs will be made.

Args:
    distance_threshold (default 0.01): Max distance a point can be from the plane model, and still be considered an inlier.

    ransac_n (default 3): Number of initial points to be considered inliers in each iteration.
    num_iterations (default 100): Maximum number of iterations.

    probability (default 0.999): Expected probability of finding the optimal plane.

Return:
    Tuple of the plane model `ax + by + cz + d = 0` and the indices of
    the plane inliers on the same device as the point cloud.

Example:

    We use Redwood dataset to compute its plane model and inliers::

        sample_pcd_data = o3d.data.PCDPointCloud()
        pcd = o3d.t.io.read_point_cloud(sample_pcd_data.path)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw([inlier_cloud, outlier_cloud]))");
    pointcloud.def(
            "compute_convex_hull", &PointCloud::ComputeConvexHull,
            "joggle_inputs"_a = false,
            R"doc(Compute the convex hull of a triangle mesh using qhull. This runs on the CPU.

Args:
    joggle_inputs (default False): Handle precision problems by randomly perturbing the input data. Set to True if perturbing the input is acceptable but you need convex simplicial output. If False, neighboring facets may be merged in case of precision problems. See `QHull docs <http://www.qhull.org/html/qh-impre.htm#joggle>`__ for more details.

Return:
    TriangleMesh representing the convexh hull. This contains an
    extra vertex property `point_indices` that contains the index of the
    corresponding vertex in the original mesh.

Example:
    We will load the Eagle dataset, compute and display it's convex hull::

        eagle = o3d.data.EaglePointCloud()
        pcd = o3d.t.io.read_point_cloud(eagle.path)
        hull = pcd.compute_convex_hull()
        o3d.visualization.draw([{'name': 'eagle', 'geometry': pcd}, {'name': 'convex hull', 'geometry': hull}])
    )doc");
    pointcloud.def("compute_boundary_points",
                   &PointCloud::ComputeBoundaryPoints, "radius"_a,
                   "max_nn"_a = 30, "angle_threshold"_a = 90.0,
                   R"doc(Compute the boundary points of a point cloud.
The implementation is inspired by the PCL implementation. Reference:
https://pointclouds.org/documentation/classpcl_1_1_boundary_estimation.html

Args:
    radius: Neighbor search radius parameter.
    max_nn (default 30): Maximum number of neighbors to search.
    angle_threshold (default 90.0): Angle threshold to decide if a point is on the boundary.

Return:
    Tensor of boundary points and its boolean mask tensor.

Example:
    We will load the DemoCropPointCloud dataset, compute its boundary points::

        ply_point_cloud = o3d.data.DemoCropPointCloud()
        pcd = o3d.t.io.read_point_cloud(ply_point_cloud.point_cloud_path)
        boundaries, mask = pcd.compute_boundary_points(radius, max_nn)
        boundaries.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw([pcd, boundaries])
    )doc");

    // conversion
    pointcloud.def("to_legacy", &PointCloud::ToLegacy,
                   "Convert to a legacy Open3D PointCloud.");
    pointcloud.def(
            "get_axis_aligned_bounding_box",
            &PointCloud::GetAxisAlignedBoundingBox,
            "Create an axis-aligned bounding box from attribute 'positions'.");
    pointcloud.def(
            "get_oriented_bounding_box", &PointCloud::GetOrientedBoundingBox,
            "Create an oriented bounding box from attribute 'positions'.");
    pointcloud.def("crop",
                   (PointCloud(PointCloud::*)(const AxisAlignedBoundingBox&,
                                              bool) const) &
                           PointCloud::Crop,
                   "Function to crop pointcloud into output pointcloud.",
                   "aabb"_a, "invert"_a = false);
    pointcloud.def("crop",
                   (PointCloud(PointCloud::*)(const OrientedBoundingBox&, bool)
                            const) &
                           PointCloud::Crop,
                   "Function to crop pointcloud into output pointcloud.",
                   "obb"_a, "invert"_a = false);

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
            m, "PointCloud", "farthest_point_down_sample",
            {{"num_samples", "Number of points to be sampled."},
             {"start_index", "Index of point to start downsampling from."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_radius_outliers",
            {{"nb_points",
              "Number of neighbor points required within the radius."},
             {"search_radius", "Radius of the sphere."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "paint_uniform_color",
            {{"color",
              "Color of the pointcloud. Floating color values are clipped "
              "between 0.0 and 1.0."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "orient_normals_to_align_with_direction",
            {{"orientation_reference",
              "Normals are oriented with respect to orientation_reference."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "orient_normals_towards_camera_location",
            {{"camera_location",
              "Normals are oriented with towards the camera_location."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "crop",
            {{"aabb", "AxisAlignedBoundingBox to crop points."},
             {"invert",
              "Crop the points outside of the bounding box or inside of the "
              "bounding box."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "crop",
            {{"obb", "OrientedBoundingBox to crop points."},
             {"invert",
              "Crop the points outside of the bounding box or inside of the "
              "bounding box."}});
    pointcloud.def("extrude_rotation", &PointCloud::ExtrudeRotation, "angle"_a,
                   "axis"_a, "resolution"_a = 16, "translation"_a = 0.0,
                   "capping"_a = true,
                   R"(Sweeps the point set rotationally about an axis.

Args:
    angle (float): The rotation angle in degree.

    axis (open3d.core.Tensor): The rotation axis.

    resolution (int): The resolution defines the number of intermediate sweeps
        about the rotation axis.

    translation (float): The translation along the rotation axis.

Returns:
    A line set with the result of the sweep operation.


Example:

    This code generates a number of helices from a point cloud::

        import open3d as o3d
        import numpy as np
        pcd = o3d.t.geometry.PointCloud(np.random.rand(10,3))
        helices = pcd.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=2)
        o3d.visualization.draw([{'name': 'helices', 'geometry': helices}])

)");

    pointcloud.def("extrude_linear", &PointCloud::ExtrudeLinear, "vector"_a,
                   "scale"_a = 1.0, "capping"_a = true,
                   R"(Sweeps the point cloud along a direction vector.

Args:

    vector (open3d.core.Tensor): The direction vector.

    scale (float): Scalar factor which essentially scales the direction vector.

Returns:
    A line set with the result of the sweep operation.


Example:

    This code generates a set of straight lines from a point cloud::

        import open3d as o3d
        import numpy as np
        pcd = o3d.t.geometry.PointCloud(np.random.rand(10,3))
        lines = pcd.extrude_linear([0,1,0])
        o3d.visualization.draw([{'name': 'lines', 'geometry': lines}])

)");

    pointcloud.def("pca_partition", &PointCloud::PCAPartition, "max_points"_a,
                   R"(Partition the point cloud by recursively doing PCA.

This function creates a new point attribute with the name "partition_ids" storing
the partition id for each point.

Args:
    max_points (int): The maximum allowed number of points in a partition.


Example:

    This code computes parititions a point cloud such that each partition
    contains at most 20 points::

        import open3d as o3d
        import numpy as np
        pcd = o3d.t.geometry.PointCloud(np.random.rand(100,3))
        num_partitions = pcd.pca_partition(max_points=20)

        # print the partition ids and the number of points for each of them.
        print(np.unique(pcd.point.partition_ids.numpy(), return_counts=True))

)");

    pointcloud.def("compute_metrics", &PointCloud::ComputeMetrics, "pcd2"_a,
                   "metrics"_a, "params"_a,
                   R"(Compute various metrics between two point clouds. 
            
Currently, Chamfer distance, Hausdorff distance and F-Score `[Knapitsch2017] <../tutorial/reference.html#Knapitsch2017>`_ are supported. 
The Chamfer distance is the sum of the mean distance to the nearest neighbor 
from the points of the first point cloud to the second point cloud. The F-Score
at a fixed threshold radius is the harmonic mean of the Precision and Recall. 
Recall is the percentage of surface points from the first point cloud that have 
the second point cloud points within the threshold radius, while Precision is 
the percentage of points from the second point cloud that have the first point 
cloud points within the threhold radius.

.. math::
    :nowrap:

    \begin{align}
        \text{Chamfer Distance: } d_{CD}(X,Y) &= \frac{1}{|X|}\sum_{i \in X} || x_i - n(x_i, Y) || + \frac{1}{|Y|}\sum_{i \in Y} || y_i - n(y_i, X) ||\\
        \text{Hausdorff distance: } d_H(X,Y) &= \max \left\{ \max_{i \in X} || x_i - n(x_i, Y) ||, \max_{i \in Y} || y_i - n(y_i, X) || \right\}\\
        \text{Precision: } P(X,Y|d) &= \frac{100}{|X|} \sum_{i \in X} || x_i - n(x_i, Y) || < d \\
        \text{Recall: } R(X,Y|d) &= \frac{100}{|Y|} \sum_{i \in Y} || y_i - n(y_i, X) || < d \\
        \text{F-Score: } F(X,Y|d) &= \frac{2 P(X,Y|d) R(X,Y|d)}{P(X,Y|d) + R(X,Y|d)} \\
    \end{align}

Args:
    pcd2 (t.geometry.PointCloud): Other point cloud to compare with.
    metrics (Sequence[t.geometry.Metric]): List of Metric s to compute. Multiple metrics can be computed at once for efficiency.
    params (t.geometry.MetricParameters): This holds parameters required by different metrics.

Returns:
    Tensor containing the requested metrics.

Example::

    from open3d.t.geometry import TriangleMesh, PointCloud, Metric, MetricParameters
    # box is a cube with one vertex at the origin and a side length 1
    pos = TriangleMesh.create_box().vertex.positions
    pcd1 = PointCloud(pos.clone())
    pcd2 = PointCloud(pos * 1.1)

    # (1, 3, 3, 1) vertices are shifted by (0, 0.1, 0.1*sqrt(2), 0.1*sqrt(3))
    # respectively
    metric_params = MetricParameters(
        fscore_radius=o3d.utility.FloatVector((0.01, 0.11, 0.15, 0.18)))
    metrics = pcd1.compute_metrics(
        pcd2, (Metric.ChamferDistance, Metric.HausdorffDistance, Metric.FScore),
        metric_params)

    print(metrics)
    np.testing.assert_allclose(
        metrics.cpu().numpy(),
        (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
        rtol=1e-6)
    )");
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
