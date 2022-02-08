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

#include "pybind/data/data.h"

#include "open3d/data/Dataset.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace data {

template <class DatasetBase = Dataset>
class PyDataset : public DatasetBase {
public:
    using DatasetBase::DatasetBase;
};

template <class SimpleDatasetBase = SingleDownloadDataset>
class PySimpleDataset : public PyDataset<SimpleDatasetBase> {
public:
    using PyDataset<SimpleDatasetBase>::PyDataset;
};

void pybind_data_classes(py::module& m) {
    // open3d.data.Dataset
    py::class_<Dataset, PyDataset<Dataset>, std::shared_ptr<Dataset>> dataset(
            m, "Dataset", "The base dataset class.");
    dataset.def(py::init<const std::string&, const std::string&>(), "prefix"_a,
                "data_root"_a = "")
            .def_property_readonly(
                    "data_root", &Dataset::GetDataRoot,
                    "Get data root directory. The data root is set at "
                    "construction time or automatically determined.")
            .def_property_readonly("prefix", &Dataset::GetPrefix,
                                   "Get prefix for the dataset.")
            .def_property_readonly(
                    "download_dir", &Dataset::GetDownloadDir,
                    "Get absolute path to download directory. i.e. "
                    "${data_root}/${download_prefix}/${prefix}")
            .def_property_readonly(
                    "extract_dir", &Dataset::GetExtractDir,
                    "Get absolute path to extract directory. i.e. "
                    "${data_root}/${extract_prefix}/${prefix}");
    docstring::ClassMethodDocInject(m, "Dataset", "data_root");
    docstring::ClassMethodDocInject(m, "Dataset", "prefix");
    docstring::ClassMethodDocInject(m, "Dataset", "download_dir");
    docstring::ClassMethodDocInject(m, "Dataset", "extract_dir");

    // open3d.data.SingleDownloadDataset
    py::class_<SingleDownloadDataset, PySimpleDataset<SingleDownloadDataset>,
               std::shared_ptr<SingleDownloadDataset>, Dataset>
            single_download_dataset(m, "SingleDownloadDataset",
                                    "Simple dataset class.");
    single_download_dataset.def(
            py::init<const std::string&, const std::vector<std::string>&,
                     const std::string&, const bool, const std::string&>(),
            "prefix"_a, "urls"_a, "md5"_a, "no_extract"_a = false,
            "data_root"_a = "",
            "This class allows user to create simple dataset which includes "
            "single file downloading and extracting / copying.");
}

void pybind_demo_icp_pointclouds(py::module& m) {
    // open3d.data.DemoICPPointClouds
    py::class_<DemoICPPointClouds, PySimpleDataset<DemoICPPointClouds>,
               std::shared_ptr<DemoICPPointClouds>, SingleDownloadDataset>
            demo_icp_pointclouds(m, "DemoICPPointClouds",
                                 "Data class for `DemoICPPointClouds` contains "
                                 "3 point clouds of binary PCD format. This "
                                 "dataset is used in Open3D for ICP demo.");
    demo_icp_pointclouds
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoICPPointClouds", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const DemoICPPointClouds& demo_icp_pointclouds) {
                        return demo_icp_pointclouds.GetPaths();
                    },
                    "List of 3 point cloud paths. Use `paths[0]`, `paths[1]`, "
                    "and `paths[2]` to access the paths.");
    docstring::ClassMethodDocInject(m, "DemoICPPointClouds", "paths");
}

void pybind_demo_colored_icp_pointclouds(py::module& m) {
    // open3d.data.DemoColoredICPPointClouds
    py::class_<DemoColoredICPPointClouds,
               PySimpleDataset<DemoColoredICPPointClouds>,
               std::shared_ptr<DemoColoredICPPointClouds>,
               SingleDownloadDataset>
            demo_colored_icp_pointclouds(
                    m, "DemoColoredICPPointClouds",
                    "Data class for `DemoColoredICPPointClouds` contains "
                    "2 point clouds of `ply` format. This dataset is used in "
                    "Open3D for colored ICP demo.");
    demo_colored_icp_pointclouds
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoColoredICPPointClouds", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const DemoColoredICPPointClouds&
                               demo_colored_icp_pointclouds) {
                        return demo_colored_icp_pointclouds.GetPaths();
                    },
                    "List of 2 point cloud paths. Use `paths[0]`, and "
                    "`paths[1]`, to access the paths.");
    docstring::ClassMethodDocInject(m, "DemoColoredICPPointClouds", "paths");
}

void pybind_demo_crop_pointcloud(py::module& m) {
    // open3d.data.DemoCropPointCloud
    py::class_<DemoCropPointCloud, PySimpleDataset<DemoCropPointCloud>,
               std::shared_ptr<DemoCropPointCloud>, SingleDownloadDataset>
            demo_crop_pointcloud(
                    m, "DemoCropPointCloud",
                    "Data class for `DemoCropPointCloud` contains a point "
                    "cloud, and `cropped.json` (a saved selected polygon "
                    "volume file). This dataset is used in Open3D for point "
                    "cloud crop demo.");
    demo_crop_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoCropPointCloud", "data_root"_a = "")
            .def_property_readonly("pointcloud_path",
                                   &DemoCropPointCloud::GetPointCloudPath,
                                   "Path to the example point cloud.")
            .def_property_readonly(
                    "cropped_json_path",
                    &DemoCropPointCloud::GetCroppedJSONPath,
                    "Path to the saved selected polygon volume file.");
    docstring::ClassMethodDocInject(m, "DemoCropPointCloud", "pointcloud_path");
    docstring::ClassMethodDocInject(m, "DemoCropPointCloud",
                                    "cropped_json_path");
}

void pybind_demo_pointcloud_feature_matching(py::module& m) {
    // open3d.data.DemoPointCloudFeatureMatching
    py::class_<DemoPointCloudFeatureMatching,
               PySimpleDataset<DemoPointCloudFeatureMatching>,
               std::shared_ptr<DemoPointCloudFeatureMatching>,
               SingleDownloadDataset>
            demo_feature_matching(
                    m, "DemoPointCloudFeatureMatching",
                    "Data class for `DemoPointCloudFeatureMatching` contains 2 "
                    "pointcloud fragments and their respective FPFH features "
                    "and L32D features. This dataset is used in Open3D for "
                    "point cloud feature matching demo.");
    demo_feature_matching
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoPointCloudFeatureMatching",
                 "data_root"_a = "")
            .def_property_readonly(
                    "pointcloud_paths",
                    &DemoPointCloudFeatureMatching::GetPointCloudPaths,
                    "List of 2 point cloud paths. Use `pointcloud_paths[0]`, "
                    "and `pointcloud_paths[1]`, to access the paths.")
            .def_property_readonly(
                    "fpfh_feature_paths",
                    &DemoPointCloudFeatureMatching::GetFPFHFeaturePaths,
                    "List of 2 saved FPFH feature binary of the respective "
                    "point cloud paths. Use `fpfh_feature_paths[0]`, "
                    "and `fpfh_feature_paths[1]`, to access the paths.")
            .def_property_readonly(
                    "l32d_feature_paths",
                    &DemoPointCloudFeatureMatching::GetL32DFeaturePaths,
                    "List of 2 saved L32D feature binary of the respective "
                    "point cloud paths. Use `l32d_feature_paths[0]`, "
                    "and `l32d_feature_paths[1]`, to access the paths.");
    docstring::ClassMethodDocInject(m, "DemoPointCloudFeatureMatching",
                                    "pointcloud_paths");
    docstring::ClassMethodDocInject(m, "DemoPointCloudFeatureMatching",
                                    "fpfh_feature_paths");
    docstring::ClassMethodDocInject(m, "DemoPointCloudFeatureMatching",
                                    "l32d_feature_paths");
}

void pybind_demo_pose_graph_optimization(py::module& m) {
    // open3d.data.DemoPoseGraphOptimization
    py::class_<DemoPoseGraphOptimization,
               PySimpleDataset<DemoPoseGraphOptimization>,
               std::shared_ptr<DemoPoseGraphOptimization>,
               SingleDownloadDataset>
            demo_pose_graph_optimization(
                    m, "DemoPoseGraphOptimization",
                    "Data class for `DemoPoseGraphOptimization` contains an "
                    "example fragment pose graph, and global pose graph. This "
                    "dataset is used in Open3D for pose graph optimization "
                    "demo.");
    demo_pose_graph_optimization
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoPoseGraphOptimization", "data_root"_a = "")
            .def_property_readonly(
                    "pose_graph_fragment_path",
                    &DemoPoseGraphOptimization::GetPoseGraphFragmentPath,
                    "Path to example global pose graph (json).")
            .def_property_readonly(
                    "pose_graph_global_path",
                    &DemoPoseGraphOptimization::GetPoseGraphGlobalPath,
                    "Path to example fragment pose graph (json).");
    docstring::ClassMethodDocInject(m, "DemoPoseGraphOptimization",
                                    "pose_graph_fragment_path");
    docstring::ClassMethodDocInject(m, "DemoPoseGraphOptimization",
                                    "pose_graph_global_path");
}

void pybind_sample_pointcloud_pcd(py::module& m) {
    // open3d.data.SamplePointCloudPCD
    py::class_<SamplePointCloudPCD, PySimpleDataset<SamplePointCloudPCD>,
               std::shared_ptr<SamplePointCloudPCD>, SingleDownloadDataset>
            pcd_pointcloud(m, "SamplePointCloudPCD",
                           "Data class for `SamplePointCloudPCD` contains the "
                           "`fragment.pcd` point cloud mesh from the `Redwood "
                           "Living Room` dataset.");
    pcd_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SamplePointCloudPCD", "data_root"_a = "")
            .def_property_readonly("path", &SamplePointCloudPCD::GetPath,
                                   "Path to the `pcd` format point cloud.");
    docstring::ClassMethodDocInject(m, "SamplePointCloudPCD", "path");
}

void pybind_sample_pointcloud_ply(py::module& m) {
    // open3d.data.SamplePointCloudPLY
    py::class_<SamplePointCloudPLY, PySimpleDataset<SamplePointCloudPLY>,
               std::shared_ptr<SamplePointCloudPLY>, SingleDownloadDataset>
            ply_pointcloud(m, "SamplePointCloudPLY",
                           "Data class for `SamplePointCloudPLY` contains the "
                           "`fragment.pcd` point cloud mesh from the `Redwood "
                           "Living Room` dataset.");
    ply_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SamplePointCloudPLY", "data_root"_a = "")
            .def_property_readonly("path", &SamplePointCloudPLY::GetPath,
                                   "Path to the `ply` format point cloud.");
    docstring::ClassMethodDocInject(m, "SamplePointCloudPLY", "path");
}

void pybind_sample_rgbd_image_nyu(py::module& m) {
    // open3d.data.SampleRGBDImageNYU
    py::class_<SampleRGBDImageNYU, PySimpleDataset<SampleRGBDImageNYU>,
               std::shared_ptr<SampleRGBDImageNYU>, SingleDownloadDataset>
            rgbd_image_nyu(m, "SampleRGBDImageNYU",
                           "Data class for `SampleRGBDImageNYU` contains a "
                           "color image `NYU_color.ppm` and a depth image "
                           "`NYU_depth.pgm` sample from NYU RGBD dataset.");
    rgbd_image_nyu
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SampleRGBDImageNYU", "data_root"_a = "")
            .def_property_readonly("color_path",
                                   &SampleRGBDImageNYU::GetColorPath,
                                   "Path to color image sample.")
            .def_property_readonly("depth_path",
                                   &SampleRGBDImageNYU::GetDepthPath,
                                   "Path to depth image sample.");
    docstring::ClassMethodDocInject(m, "SampleRGBDImageNYU", "color_path");
    docstring::ClassMethodDocInject(m, "SampleRGBDImageNYU", "depth_path");
}

void pybind_sample_rgbd_image_sun(py::module& m) {
    // open3d.data.SampleRGBDImageSUN
    py::class_<SampleRGBDImageSUN, PySimpleDataset<SampleRGBDImageSUN>,
               std::shared_ptr<SampleRGBDImageSUN>, SingleDownloadDataset>
            rgbd_image_sun(m, "SampleRGBDImageSUN",
                           "Data class for `SampleRGBDImageSUN` contains a "
                           "color image `SUN_color.jpg` and a depth image "
                           "`SUN_depth.png` sample from SUN RGBD dataset.");
    rgbd_image_sun
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SampleRGBDImageSUN", "data_root"_a = "")
            .def_property_readonly("color_path",
                                   &SampleRGBDImageSUN::GetColorPath,
                                   "Path to color image sample.")
            .def_property_readonly("depth_path",
                                   &SampleRGBDImageSUN::GetDepthPath,
                                   "Path to depth image sample.");
    docstring::ClassMethodDocInject(m, "SampleRGBDImageSUN", "color_path");
    docstring::ClassMethodDocInject(m, "SampleRGBDImageSUN", "depth_path");
}

void pybind_sample_rgbd_image_tum(py::module& m) {
    // open3d.data.SampleRGBDImageTUM
    py::class_<SampleRGBDImageTUM, PySimpleDataset<SampleRGBDImageTUM>,
               std::shared_ptr<SampleRGBDImageTUM>, SingleDownloadDataset>
            rgbd_image_tum(m, "SampleRGBDImageTUM",
                           "Data class for `SampleRGBDImageTUM` contains a "
                           "color image `TUM_color.png` and a depth image "
                           "`TUM_depth.png` sample from TUM RGBD dataset.");
    rgbd_image_tum
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SampleRGBDImageTUM", "data_root"_a = "")
            .def_property_readonly("color_path",
                                   &SampleRGBDImageTUM::GetColorPath,
                                   "Path to color image sample.")
            .def_property_readonly("depth_path",
                                   &SampleRGBDImageTUM::GetDepthPath,
                                   "Path to depth image sample.");
    docstring::ClassMethodDocInject(m, "SampleRGBDImageTUM", "color_path");
    docstring::ClassMethodDocInject(m, "SampleRGBDImageTUM", "depth_path");
}

void pybind_sample_rgbd_dataset_redwood(py::module& m) {
    // open3d.data.SampleRGBDDatasetRedwood
    py::class_<SampleRGBDDatasetRedwood,
               PySimpleDataset<SampleRGBDDatasetRedwood>,
               std::shared_ptr<SampleRGBDDatasetRedwood>, SingleDownloadDataset>
            rgbd_dataset_icl(
                    m, "SampleRGBDDatasetRedwood",
                    "Data class for `SampleRGBDDatasetICL` contains a sample "
                    "set of 5 color and depth images from Redwood RGBD "
                    "dataset living-room1. Additionally it also contains "
                    "camera trajectory log, camera odometry log, RGBD match, "
                    "and point cloud reconstruction from TSDF.");
    rgbd_dataset_icl
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SampleRGBDDatasetRedwood", "data_root"_a = "")
            .def_property_readonly(
                    "color_paths", &SampleRGBDDatasetRedwood::GetColorPaths,
                    "List of paths to color image samples of size 5. Use "
                    "`color_paths[0]`, `color_paths[1]` ... `color_paths[4]` "
                    "to access the paths.")
            .def_property_readonly(
                    "depth_paths", &SampleRGBDDatasetRedwood::GetDepthPaths,
                    "List of paths to depth image samples of size 5. Use "
                    "`depth_paths[0]`, `depth_paths[1]` ... `depth_paths[4]` "
                    "to access the paths.")
            .def_property_readonly(
                    "trajectory_log_path",
                    &SampleRGBDDatasetRedwood::GetTrajectoryLogPath,
                    "Path to camera trajectory log file `trajectory.log`.")
            .def_property_readonly(
                    "odometry_log_path",
                    &SampleRGBDDatasetRedwood::GetOdometryLogPath,
                    "Path to camera odometry log file `odometry.log`.")
            .def_property_readonly(
                    "rgbd_match_path",
                    &SampleRGBDDatasetRedwood::GetRGBDMatchPath,
                    "Path to color and depth image match file `rgbd.match`.")
            .def_property_readonly(
                    "reconstruction_path",
                    &SampleRGBDDatasetRedwood::GetReconstructionPath,
                    "Path to pointcloud reconstruction from TSDF.");
    docstring::ClassMethodDocInject(m, "SampleRGBDDatasetRedwood",
                                    "color_paths");
    docstring::ClassMethodDocInject(m, "SampleRGBDDatasetRedwood",
                                    "depth_paths");
    docstring::ClassMethodDocInject(m, "SampleRGBDDatasetRedwood",
                                    "trajectory_log_path");
    docstring::ClassMethodDocInject(m, "SampleRGBDDatasetRedwood",
                                    "odometry_log_path");
    docstring::ClassMethodDocInject(m, "SampleRGBDDatasetRedwood",
                                    "rgbd_match_path");
    docstring::ClassMethodDocInject(m, "SampleRGBDDatasetRedwood",
                                    "reconstruction_path");
}

void pybind_sample_fountain_rgbd_dataset(py::module& m) {
    // open3d.data.SampleFountainRGBDDataset
    py::class_<SampleFountainRGBDDataset,
               PySimpleDataset<SampleFountainRGBDDataset>,
               std::shared_ptr<SampleFountainRGBDDataset>,
               SingleDownloadDataset>
            fountain_rgbd_dataset(
                    m, "SampleFountainRGBDDataset",
                    "Data class for `SampleFountainRGBDDataset` contains a "
                    "sample set of 33 color and depth images from the "
                    "`Fountain RGBD dataset`. It also contains `camera poses "
                    "at keyframes log` and `mesh reconstruction`. It is used "
                    "in demo of `Color Map Optimization`.");
    fountain_rgbd_dataset
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SampleFountainRGBDDataset", "data_root"_a = "")
            .def_property_readonly(
                    "color_paths", &SampleFountainRGBDDataset::GetColorPaths,
                    "List of paths to color image samples of size 33. Use "
                    "`color_paths[0]`, `color_paths[1]` ... `color_paths[32]` "
                    "to access the paths.")
            .def_property_readonly(
                    "depth_paths", &SampleFountainRGBDDataset::GetDepthPaths,
                    "List of paths to depth image samples of size 33. Use "
                    "`depth_paths[0]`, `depth_paths[1]` ... `depth_paths[32]` "
                    "to access the paths.")
            .def_property_readonly(
                    "keyframe_poses_log_path",
                    &SampleFountainRGBDDataset::GetKeyframePosesLogPath,
                    "Path to camera poses at keyfragmes log file `key.log`.")
            .def_property_readonly(
                    "reconstruction_path",
                    &SampleFountainRGBDDataset::GetReconstructionPath,
                    "Path to mesh reconstruction.");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDDataset",
                                    "color_paths");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDDataset",
                                    "depth_paths");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDDataset",
                                    "keyframe_poses_log_path");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDDataset",
                                    "reconstruction_path");
}

void pybind_eagle(py::module& m) {
    // open3d.data.EaglePointCloud
    py::class_<EaglePointCloud, PySimpleDataset<EaglePointCloud>,
               std::shared_ptr<EaglePointCloud>, SingleDownloadDataset>
            eagle(m, "EaglePointCloud",
                  "Data class for `EaglePointCloud` contains the "
                  "`EaglePointCloud.ply` "
                  "file.");
    eagle.def(py::init<const std::string&, const std::string&>(),
              "prefix"_a = "EaglePointCloud", "data_root"_a = "")
            .def_property_readonly("path", &EaglePointCloud::GetPath,
                                   "Path to the `EaglePointCloud.ply` file.");
    docstring::ClassMethodDocInject(m, "EaglePointCloud", "path");
}

void pybind_armadillo(py::module& m) {
    // open3d.data.ArmadilloMesh
    py::class_<ArmadilloMesh, PySimpleDataset<ArmadilloMesh>,
               std::shared_ptr<ArmadilloMesh>, SingleDownloadDataset>
            armadillo(m, "ArmadilloMesh",
                      "Data class for `ArmadilloMesh` contains the "
                      "`ArmadilloMesh.ply` from the `Stanford 3D Scanning "
                      "Repository`.");
    armadillo
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "ArmadilloMesh", "data_root"_a = "")
            .def_property_readonly("path", &ArmadilloMesh::GetPath,
                                   "Path to the `ArmadilloMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "ArmadilloMesh", "path");
}

void pybind_bunny(py::module& m) {
    // open3d.data.BunnyMesh
    py::class_<BunnyMesh, PySimpleDataset<BunnyMesh>,
               std::shared_ptr<BunnyMesh>, SingleDownloadDataset>
            bunny(m, "BunnyMesh",
                  "Data class for `BunnyMesh` contains the `BunnyMesh.ply` "
                  "from "
                  "the `Stanford 3D Scanning Repository`.");
    bunny.def(py::init<const std::string&, const std::string&>(),
              "prefix"_a = "BunnyMesh", "data_root"_a = "")
            .def_property_readonly("path", &BunnyMesh::GetPath,
                                   "Path to the `BunnyMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "BunnyMesh", "path");
}

void pybind_knot(py::module& m) {
    // open3d.data.KnotMesh
    py::class_<KnotMesh, PySimpleDataset<KnotMesh>, std::shared_ptr<KnotMesh>,
               SingleDownloadDataset>
            knot(m, "KnotMesh",
                 "Data class for `KnotMesh` contains the `KnotMesh.ply`.");
    knot.def(py::init<const std::string&, const std::string&>(),
             "prefix"_a = "KnotMesh", "data_root"_a = "")
            .def_property_readonly("path", &KnotMesh::GetPath,
                                   "Path to the `KnotMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "KnotMesh", "path");
}

void pybind_juneau(py::module& m) {
    // open3d.data.JuneauImage
    py::class_<JuneauImage, PySimpleDataset<JuneauImage>,
               std::shared_ptr<JuneauImage>, SingleDownloadDataset>
            juneau(m, "JuneauImage",
                   "Data class for `JuneauImage` contains the "
                   "`JuneauImage.jpg` "
                   "file.");
    juneau.def(py::init<const std::string&, const std::string&>(),
               "prefix"_a = "JuneauImage", "data_root"_a = "")
            .def_property_readonly("path", &JuneauImage::GetPath,
                                   "Path to the `JuneauImage.jgp` file.");
    docstring::ClassMethodDocInject(m, "JuneauImage", "path");
}

void pybind_redwood_living_room_pointclouds(py::module& m) {
    // open3d.data.RedwoodLivingRoomPointClouds
    py::class_<RedwoodLivingRoomPointClouds,
               PySimpleDataset<RedwoodLivingRoomPointClouds>,
               std::shared_ptr<RedwoodLivingRoomPointClouds>,
               SingleDownloadDataset>
            redwood_living_room_pointclouds(
                    m, "RedwoodLivingRoomPointClouds",
                    "Dataset class for `RedwoodLivingRoomPointClouds` contains "
                    "57 point clouds of binary PLY format.");
    redwood_living_room_pointclouds
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "RedwoodLivingRoomPointClouds",
                 "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const RedwoodLivingRoomPointClouds&
                               redwood_living_room_pointclouds) {
                        return redwood_living_room_pointclouds.GetPaths();
                    },
                    "List of paths to ply point-cloud fragments of size 57. "
                    "Use `paths[0]`, `paths[1]` ... `paths[56]` to access the "
                    "paths.");
    docstring::ClassMethodDocInject(m, "RedwoodLivingRoomPointClouds", "paths");
}

void pybind_redwood_office_pointclouds(py::module& m) {
    // open3d.data.RedwoodOfficePointClouds
    py::class_<RedwoodOfficePointClouds,
               PySimpleDataset<RedwoodOfficePointClouds>,
               std::shared_ptr<RedwoodOfficePointClouds>, SingleDownloadDataset>
            redwood_office_pointclouds(
                    m, "RedwoodOfficePointClouds",
                    "Dataset class for `RedwoodOfficePointClouds` contains 53 "
                    "point clouds of binary PLY format.");
    redwood_office_pointclouds
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "RedwoodOfficePointClouds", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const RedwoodOfficePointClouds&
                               redwood_office_pointclouds) {
                        return redwood_office_pointclouds.GetPaths();
                    },
                    "List of paths to ply point-cloud fragments of size 53. "
                    "Use `paths[0]`, `paths[1]` ... `paths[52]` to access the "
                    "paths.");
    docstring::ClassMethodDocInject(m, "RedwoodOfficePointClouds", "paths");
}

void pybind_data(py::module& m) {
    py::module m_submodule = m.def_submodule("data", "Data handling module.");
    pybind_data_classes(m_submodule);

    pybind_demo_icp_pointclouds(m_submodule);
    pybind_demo_colored_icp_pointclouds(m_submodule);
    pybind_demo_crop_pointcloud(m_submodule);
    pybind_demo_pointcloud_feature_matching(m_submodule);
    pybind_demo_pose_graph_optimization(m_submodule);
    pybind_sample_pointcloud_pcd(m_submodule);
    pybind_sample_pointcloud_ply(m_submodule);
    pybind_sample_rgbd_image_nyu(m_submodule);
    pybind_sample_rgbd_image_sun(m_submodule);
    pybind_sample_rgbd_image_tum(m_submodule);
    pybind_sample_rgbd_dataset_redwood(m_submodule);
    pybind_sample_fountain_rgbd_dataset(m_submodule);
    pybind_eagle(m_submodule);
    pybind_armadillo(m_submodule);
    pybind_bunny(m_submodule);
    pybind_knot(m_submodule);
    pybind_juneau(m_submodule);
    pybind_redwood_living_room_pointclouds(m_submodule);
    pybind_redwood_office_pointclouds(m_submodule);
}

}  // namespace data
}  // namespace open3d
