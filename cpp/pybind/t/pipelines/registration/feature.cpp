// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Feature.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"
#include "pybind/t/pipelines/registration/registration.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

void pybind_feature_definitions(py::module &m_registration) {
    m_registration.def("compute_fpfh_feature", &ComputeFPFHFeature,
                       py::call_guard<py::gil_scoped_release>(), "input"_a,
                       "max_nn"_a = 100, "radius"_a = py::none(),
                       "indices"_a = py::none(),
                       R"(Function to compute FPFH feature for a point cloud.

It uses KNN search (Not recommended to use on GPU) if only max_nn parameter
is provided, Radius search (Not recommended to use on GPU) if only radius
parameter is provided, and Hybrid search (Recommended) if both are provided.
If indices is provided, the function will compute FPFH features only on the
selected points.

Args:
    input (open3d.core.Tensor): The input point cloud with data type float32 or float64.
    max_nn (int, optional): Neighbor search max neighbors parameter.
        Default is 100.
    radius (float, optional): Neighbor search radius parameter.
        Recommended value is ~5x voxel size.
    indices (open3d.core.Tensor, optional): Tensor with the indices of the points to
        compute FPFH features on. Default is None.

Returns:
    The FPFH feature tensor with shape (N, 33).

Example:
    This example shows how to compute the FPFH features for a point cloud::

        import open3d as o3d
        # read and downsample point clouds
        paths = o3d.data.DemoICPPointClouds().paths
        voxel_size = 0.01
        pcd = o3d.t.io.read_point_cloud(paths[0]).voxel_down_sample(voxel_size)

        # compute FPFH features
        pcd_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(pcd, radius=5*voxel_size)
    )"

    );

    m_registration.def(
            "correspondences_from_features", &CorrespondencesFromFeatures,
            py::call_guard<py::gil_scoped_release>(), "source_features"_a,
            "target_features"_a, "mutual_filter"_a = false,
            "mutual_consistency_ratio"_a = 0.1f,
            R"(Function to query nearest neighbors of source_features in target_features.
            
Args:
    source_features (open3d.core.Tensor): The source features in shape (N, dim).
    target_features (open3d.core.Tensor): The target features in shape (M, dim).
    mutual_filter (bool, optional): Filter correspondences and return the 
        collection of (i, j) s.t. source_features[i] and target_features[j] are 
        mutually the nearest neighbor. Default is False.
    mutual_consistency_ratio (float, optional): Threshold to decide whether the 
        number of filtered correspondences is sufficient. Only used when 
        `mutual_filter` is enabled. Default is 0.1.

Returns:
    Tensor with shape (K,2) of source_indices and target_indices with K as the 
    number of correspondences.

Example:

    This example shows how to compute the features and correspondences for two point clouds::

        import open3d as o3d

        # read and downsample point clouds
        paths = o3d.data.DemoICPPointClouds().paths
        voxel_size = 0.01
        pcd1 = o3d.t.io.read_point_cloud(paths[0]).voxel_down_sample(voxel_size)
        pcd2 = o3d.t.io.read_point_cloud(paths[1]).voxel_down_sample(voxel_size)

        # compute FPFH features
        pcd1_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(pcd1, radius=5*voxel_size)
        pcd2_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(pcd2, radius=5*voxel_size)

        # compute correspondences
        matches = o3d.t.pipelines.registration.correspondences_from_features(pcd1_fpfh, pcd2_fpfh, mutual_filter=True)

        # visualize correspondences
        matches = matches[::500]
        pcd2.translate([0,2,0]) # translate pcd2 for the visualization
        lines = o3d.t.geometry.LineSet()
        lines.point.positions = o3d.core.Tensor.zeros((matches.num_elements(), 3))
        lines.point.positions[0::2] = pcd1.point.positions[matches[:,0]]
        lines.point.positions[1::2] = pcd2.point.positions[matches[:,1]]
        lines.line.indices = o3d.core.Tensor.arange(matches.num_elements()).reshape((-1,2))

        o3d.visualization.draw([pcd1, pcd2, lines])

)");
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
