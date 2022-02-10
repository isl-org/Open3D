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
    demo_icp_pointclouds.def(py::init<const std::string&>(), "data_root"_a = "")
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
            .def(py::init<const std::string&>(), "data_root"_a = "")
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
    demo_crop_pointcloud.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("point_cloud_path",
                                   &DemoCropPointCloud::GetPointCloudPath,
                                   "Path to the example point cloud.")
            .def_property_readonly(
                    "cropped_json_path",
                    &DemoCropPointCloud::GetCroppedJSONPath,
                    "Path to the saved selected polygon volume file.");
    docstring::ClassMethodDocInject(m, "DemoCropPointCloud",
                                    "point_cloud_path");
    docstring::ClassMethodDocInject(m, "DemoCropPointCloud",
                                    "cropped_json_path");
}

void pybind_demo_feature_matching_point_clouds(py::module& m) {
    // open3d.data.DemoFeatureMatchingPointClouds
    py::class_<DemoFeatureMatchingPointClouds,
               PySimpleDataset<DemoFeatureMatchingPointClouds>,
               std::shared_ptr<DemoFeatureMatchingPointClouds>,
               SingleDownloadDataset>
            demo_feature_matching(
                    m, "DemoFeatureMatchingPointClouds",
                    "Data class for `DemoFeatureMatchingPointClouds` contains "
                    "2 "
                    "pointcloud fragments and their respective FPFH features "
                    "and L32D features. This dataset is used in Open3D for "
                    "point cloud feature matching demo.");
    demo_feature_matching
            .def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly(
                    "point_cloud_paths",
                    &DemoFeatureMatchingPointClouds::GetPointCloudPaths,
                    "List of 2 point cloud paths. Use `point_cloud_paths[0]`, "
                    "and `point_cloud_paths[1]`, to access the paths.")
            .def_property_readonly(
                    "fpfh_feature_paths",
                    &DemoFeatureMatchingPointClouds::GetFPFHFeaturePaths,
                    "List of 2 saved FPFH feature binary of the respective "
                    "point cloud paths. Use `fpfh_feature_paths[0]`, "
                    "and `fpfh_feature_paths[1]`, to access the paths.")
            .def_property_readonly(
                    "l32d_feature_paths",
                    &DemoFeatureMatchingPointClouds::GetL32DFeaturePaths,
                    "List of 2 saved L32D feature binary of the respective "
                    "point cloud paths. Use `l32d_feature_paths[0]`, "
                    "and `l32d_feature_paths[1]`, to access the paths.");
    docstring::ClassMethodDocInject(m, "DemoFeatureMatchingPointClouds",
                                    "point_cloud_paths");
    docstring::ClassMethodDocInject(m, "DemoFeatureMatchingPointClouds",
                                    "fpfh_feature_paths");
    docstring::ClassMethodDocInject(m, "DemoFeatureMatchingPointClouds",
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
            .def(py::init<const std::string&>(), "data_root"_a = "")
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

void pybind_pcd_point_cloud(py::module& m) {
    // open3d.data.PCDPointCloud
    py::class_<PCDPointCloud, PySimpleDataset<PCDPointCloud>,
               std::shared_ptr<PCDPointCloud>, SingleDownloadDataset>
            pcd_pointcloud(m, "PCDPointCloud",
                           "Data class for `PCDPointCloud` contains the "
                           "`fragment.pcd` point cloud mesh from the `Redwood "
                           "Living Room` dataset.");
    pcd_pointcloud.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("path", &PCDPointCloud::GetPath,
                                   "Path to the `pcd` format point cloud.");
    docstring::ClassMethodDocInject(m, "PCDPointCloud", "path");
}

void pybind_ply_point_cloud(py::module& m) {
    // open3d.data.PLYPointCloud
    py::class_<PLYPointCloud, PySimpleDataset<PLYPointCloud>,
               std::shared_ptr<PLYPointCloud>, SingleDownloadDataset>
            ply_pointcloud(m, "PLYPointCloud",
                           "Data class for `PLYPointCloud` contains the "
                           "`fragment.pcd` point cloud mesh from the `Redwood "
                           "Living Room` dataset.");
    ply_pointcloud.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("path", &PLYPointCloud::GetPath,
                                   "Path to the `ply` format point cloud.");
    docstring::ClassMethodDocInject(m, "PLYPointCloud", "path");
}

void pybind_sample_nyu_rgbd_image(py::module& m) {
    // open3d.data.SampleNYURGBDImage
    py::class_<SampleNYURGBDImage, PySimpleDataset<SampleNYURGBDImage>,
               std::shared_ptr<SampleNYURGBDImage>, SingleDownloadDataset>
            rgbd_image_nyu(m, "SampleNYURGBDImage",
                           "Data class for `SampleNYURGBDImage` contains a "
                           "color image `NYU_color.ppm` and a depth image "
                           "`NYU_depth.pgm` sample from NYU RGBD dataset.");
    rgbd_image_nyu.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("color_path",
                                   &SampleNYURGBDImage::GetColorPath,
                                   "Path to color image sample.")
            .def_property_readonly("depth_path",
                                   &SampleNYURGBDImage::GetDepthPath,
                                   "Path to depth image sample.");
    docstring::ClassMethodDocInject(m, "SampleNYURGBDImage", "color_path");
    docstring::ClassMethodDocInject(m, "SampleNYURGBDImage", "depth_path");
}

void pybind_sample_sun_rgbd_image(py::module& m) {
    // open3d.data.SampleSUNRGBDImage
    py::class_<SampleSUNRGBDImage, PySimpleDataset<SampleSUNRGBDImage>,
               std::shared_ptr<SampleSUNRGBDImage>, SingleDownloadDataset>
            rgbd_image_sun(m, "SampleSUNRGBDImage",
                           "Data class for `SampleSUNRGBDImage` contains a "
                           "color image `SUN_color.jpg` and a depth image "
                           "`SUN_depth.png` sample from SUN RGBD dataset.");
    rgbd_image_sun.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("color_path",
                                   &SampleSUNRGBDImage::GetColorPath,
                                   "Path to color image sample.")
            .def_property_readonly("depth_path",
                                   &SampleSUNRGBDImage::GetDepthPath,
                                   "Path to depth image sample.");
    docstring::ClassMethodDocInject(m, "SampleSUNRGBDImage", "color_path");
    docstring::ClassMethodDocInject(m, "SampleSUNRGBDImage", "depth_path");
}

void pybind_sample_tum_rgbd_image(py::module& m) {
    // open3d.data.SampleTUMRGBDImage
    py::class_<SampleTUMRGBDImage, PySimpleDataset<SampleTUMRGBDImage>,
               std::shared_ptr<SampleTUMRGBDImage>, SingleDownloadDataset>
            rgbd_image_tum(m, "SampleTUMRGBDImage",
                           "Data class for `SampleTUMRGBDImage` contains a "
                           "color image `TUM_color.png` and a depth image "
                           "`TUM_depth.png` sample from TUM RGBD dataset.");
    rgbd_image_tum.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("color_path",
                                   &SampleTUMRGBDImage::GetColorPath,
                                   "Path to color image sample.")
            .def_property_readonly("depth_path",
                                   &SampleTUMRGBDImage::GetDepthPath,
                                   "Path to depth image sample.");
    docstring::ClassMethodDocInject(m, "SampleTUMRGBDImage", "color_path");
    docstring::ClassMethodDocInject(m, "SampleTUMRGBDImage", "depth_path");
}

void pybind_sample_redwood_rgbd_images(py::module& m) {
    // open3d.data.SampleRedwoodRGBDImages
    py::class_<SampleRedwoodRGBDImages,
               PySimpleDataset<SampleRedwoodRGBDImages>,
               std::shared_ptr<SampleRedwoodRGBDImages>, SingleDownloadDataset>
            rgbd_dataset_redwood(
                    m, "SampleRedwoodRGBDImages",
                    "Data class for `SampleRedwoodRGBDImages` contains a "
                    "sample set of 5 color and depth images from Redwood RGBD "
                    "dataset living-room1. Additionally it also contains "
                    "camera trajectory log, camera odometry log, rgbd match, "
                    "and point cloud reconstruction obtained using TSDF.");
    rgbd_dataset_redwood.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly(
                    "color_paths", &SampleRedwoodRGBDImages::GetColorPaths,
                    "List of paths to color image samples of size 5. Use "
                    "`color_paths[0]`, `color_paths[1]` ... `color_paths[4]` "
                    "to access the paths.")
            .def_property_readonly(
                    "depth_paths", &SampleRedwoodRGBDImages::GetDepthPaths,
                    "List of paths to depth image samples of size 5. Use "
                    "`depth_paths[0]`, `depth_paths[1]` ... `depth_paths[4]` "
                    "to access the paths.")
            .def_property_readonly(
                    "trajectory_log_path",
                    &SampleRedwoodRGBDImages::GetTrajectoryLogPath,
                    "Path to camera trajectory log file `trajectory.log`.")
            .def_property_readonly(
                    "odometry_log_path",
                    &SampleRedwoodRGBDImages::GetOdometryLogPath,
                    "Path to camera odometry log file `odometry.log`.")
            .def_property_readonly(
                    "rgbd_match_path",
                    &SampleRedwoodRGBDImages::GetRGBDMatchPath,
                    "Path to color and depth image match file `rgbd.match`.")
            .def_property_readonly(
                    "reconstruction_path",
                    &SampleRedwoodRGBDImages::GetReconstructionPath,
                    "Path to pointcloud reconstruction from TSDF.");
    docstring::ClassMethodDocInject(m, "SampleRedwoodRGBDImages",
                                    "color_paths");
    docstring::ClassMethodDocInject(m, "SampleRedwoodRGBDImages",
                                    "depth_paths");
    docstring::ClassMethodDocInject(m, "SampleRedwoodRGBDImages",
                                    "trajectory_log_path");
    docstring::ClassMethodDocInject(m, "SampleRedwoodRGBDImages",
                                    "odometry_log_path");
    docstring::ClassMethodDocInject(m, "SampleRedwoodRGBDImages",
                                    "rgbd_match_path");
    docstring::ClassMethodDocInject(m, "SampleRedwoodRGBDImages",
                                    "reconstruction_path");
}

void pybind_sample_fountain_rgbd_images(py::module& m) {
    // open3d.data.SampleFountainRGBDImages
    py::class_<SampleFountainRGBDImages,
               PySimpleDataset<SampleFountainRGBDImages>,
               std::shared_ptr<SampleFountainRGBDImages>, SingleDownloadDataset>
            fountain_rgbd_dataset(
                    m, "SampleFountainRGBDImages",
                    "Data class for `SampleFountainRGBDImages` contains a "
                    "sample set of 33 color and depth images from the "
                    "`Fountain RGBD dataset`. It also contains `camera poses "
                    "at keyframes log` and `mesh reconstruction`. It is used "
                    "in demo of `Color Map Optimization`.");
    fountain_rgbd_dataset
            .def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly(
                    "color_paths", &SampleFountainRGBDImages::GetColorPaths,
                    "List of paths to color image samples of size 33. Use "
                    "`color_paths[0]`, `color_paths[1]` ... `color_paths[32]` "
                    "to access the paths.")
            .def_property_readonly(
                    "depth_paths", &SampleFountainRGBDImages::GetDepthPaths,
                    "List of paths to depth image samples of size 33. Use "
                    "`depth_paths[0]`, `depth_paths[1]` ... `depth_paths[32]` "
                    "to access the paths.")
            .def_property_readonly(
                    "keyframe_poses_log_path",
                    &SampleFountainRGBDImages::GetKeyframePosesLogPath,
                    "Path to camera poses at keyfragmes log file `key.log`.")
            .def_property_readonly(
                    "reconstruction_path",
                    &SampleFountainRGBDImages::GetReconstructionPath,
                    "Path to mesh reconstruction.");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDImages",
                                    "color_paths");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDImages",
                                    "depth_paths");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDImages",
                                    "keyframe_poses_log_path");
    docstring::ClassMethodDocInject(m, "SampleFountainRGBDImages",
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
    eagle.def(py::init<const std::string&>(), "data_root"_a = "")
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
    armadillo.def(py::init<const std::string&>(), "data_root"_a = "")
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
    bunny.def(py::init<const std::string&>(), "data_root"_a = "")
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
    knot.def(py::init<const std::string&>(), "data_root"_a = "")
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
    juneau.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly("path", &JuneauImage::GetPath,
                                   "Path to the `JuneauImage.jgp` file.");
    docstring::ClassMethodDocInject(m, "JuneauImage", "path");
}

void pybind_living_room_point_clouds(py::module& m) {
    // open3d.data.LivingRoomPointClouds
    py::class_<LivingRoomPointClouds, PySimpleDataset<LivingRoomPointClouds>,
               std::shared_ptr<LivingRoomPointClouds>, SingleDownloadDataset>
            living_room_point_clouds(
                    m, "LivingRoomPointClouds",
                    "Dataset class for `LivingRoomPointClouds` contains "
                    "57 point clouds of binary PLY format.");
    living_room_point_clouds
            .def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const LivingRoomPointClouds& living_room_point_clouds) {
                        return living_room_point_clouds.GetPaths();
                    },
                    "List of paths to ply point-cloud fragments of size 57. "
                    "Use `paths[0]`, `paths[1]` ... `paths[56]` to access the "
                    "paths.");
    docstring::ClassMethodDocInject(m, "LivingRoomPointClouds", "paths");
}

void pybind_office_point_clouds(py::module& m) {
    // open3d.data.OfficePointClouds
    py::class_<OfficePointClouds, PySimpleDataset<OfficePointClouds>,
               std::shared_ptr<OfficePointClouds>, SingleDownloadDataset>
            office_point_clouds(
                    m, "OfficePointClouds",
                    "Dataset class for `OfficePointClouds` contains 53 "
                    "point clouds of binary PLY format.");
    office_point_clouds.def(py::init<const std::string&>(), "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const OfficePointClouds& office_point_clouds) {
                        return office_point_clouds.GetPaths();
                    },
                    "List of paths to ply point-cloud fragments of size 53. "
                    "Use `paths[0]`, `paths[1]` ... `paths[52]` to access the "
                    "paths.");
    docstring::ClassMethodDocInject(m, "OfficePointClouds", "paths");
}

void pybind_data(py::module& m) {
    py::module m_submodule = m.def_submodule("data", "Data handling module.");
    pybind_data_classes(m_submodule);

    pybind_demo_icp_pointclouds(m_submodule);
    pybind_demo_colored_icp_pointclouds(m_submodule);
    pybind_demo_crop_pointcloud(m_submodule);
    pybind_demo_feature_matching_point_clouds(m_submodule);
    pybind_demo_pose_graph_optimization(m_submodule);
    pybind_pcd_point_cloud(m_submodule);
    pybind_ply_point_cloud(m_submodule);
    pybind_sample_nyu_rgbd_image(m_submodule);
    pybind_sample_sun_rgbd_image(m_submodule);
    pybind_sample_tum_rgbd_image(m_submodule);
    pybind_sample_redwood_rgbd_images(m_submodule);
    pybind_sample_fountain_rgbd_images(m_submodule);
    pybind_eagle(m_submodule);
    pybind_armadillo(m_submodule);
    pybind_bunny(m_submodule);
    pybind_knot(m_submodule);
    pybind_juneau(m_submodule);
    pybind_living_room_point_clouds(m_submodule);
    pybind_office_point_clouds(m_submodule);
}

}  // namespace data
}  // namespace open3d
