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

template <class SimpleDatasetBase = SimpleDataset>
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
            .def_property_readonly("data_root", &Dataset::GetDataRoot,
                                   "Returns data root path.")
            .def_property_readonly("prefix", &Dataset::GetPrefix,
                                   "Returns data prefix.")
            .def_property_readonly("download_dir", &Dataset::GetDownloadDir,
                                   "Returns data download path.")
            .def_property_readonly("extract_dir", &Dataset::GetExtractDir,
                                   "Returns data extract path.");
    docstring::ClassMethodDocInject(m, "Dataset", "data_root");
    docstring::ClassMethodDocInject(m, "Dataset", "prefix");
    docstring::ClassMethodDocInject(m, "Dataset", "download_dir");
    docstring::ClassMethodDocInject(m, "Dataset", "extract_dir");

    // open3d.data.SimpleDataset
    py::class_<SimpleDataset, PySimpleDataset<SimpleDataset>,
               std::shared_ptr<SimpleDataset>, Dataset>
            simple_dataset(m, "SimpleDataset", "Simple dataset class.");
    simple_dataset.def(
            py::init<const std::string&, const std::vector<std::string>&,
                     const std::string&, const bool, const std::string&>(),
            "prefix"_a, "urls"_a, "md5"_a, "no_extract"_a = false,
            "data_root"_a = "");
}

void pybind_demo_icp_pointclouds(py::module& m) {
    // open3d.data.DemoICPPointClouds
    py::class_<DemoICPPointClouds, PySimpleDataset<DemoICPPointClouds>,
               std::shared_ptr<DemoICPPointClouds>, SimpleDataset>
            demo_icp_pointclouds(m, "DemoICPPointClouds",
                                 "Data class for `DemoICPPointClouds` contains "
                                 "3 `pointclouds` of `pcd binary` format. This "
                                 "dataset is used in Open3D for ICP demo.");
    demo_icp_pointclouds
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoICPPointClouds", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const DemoICPPointClouds& demo_icp_pointclouds) {
                        return demo_icp_pointclouds.GetPaths();
                    },
                    "List of 3 point cloud paths. Use `paths_[0]`, "
                    "`paths_[1]`, and `paths_[2]` to access the paths.");
    docstring::ClassMethodDocInject(m, "DemoICPPointClouds", "paths");
}

void pybind_demo_colored_icp_pointclouds(py::module& m) {
    // open3d.data.DemoColoredICPPointClouds
    py::class_<DemoColoredICPPointClouds,
               PySimpleDataset<DemoColoredICPPointClouds>,
               std::shared_ptr<DemoColoredICPPointClouds>, SimpleDataset>
            demo_colored_icp_pointclouds(
                    m, "DemoColoredICPPointClouds",
                    "Data class for `DemoColoredICPPointClouds` contains "
                    "2 `pointclouds` of `ply` format. This dataset is used in "
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
                    "List of 2 point cloud paths. Use `paths_[0]`, "
                    "and `paths_[1]` to access the paths.");
    docstring::ClassMethodDocInject(m, "DemoColoredICPPointClouds", "paths");
}

void pybind_demo_crop_pointcloud(py::module& m) {
    // open3d.data.DemoCropPointCloud
    py::class_<DemoCropPointCloud, PySimpleDataset<DemoCropPointCloud>,
               std::shared_ptr<DemoCropPointCloud>, SimpleDataset>
            demo_crop_pointcloud(
                    m, "DemoCropPointCloud",
                    "Data class for `DemoCropPointCloud` contains a point "
                    "cloud, and `cropped.json` (a saved selected polygon "
                    "volume file). This dataset is used in Open3D for point "
                    "cloud crop demo.");
    demo_crop_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "DemoCropPointCloud", "data_root"_a = "")
            .def_property_readonly(
                    "path_pointcloud",
                    [](const DemoCropPointCloud& demo_crop_pointcloud) {
                        return demo_crop_pointcloud.GetPathPointCloud();
                    },
                    "Path to example point cloud.")
            .def_property_readonly(
                    "path_cropped_json",
                    [](const DemoCropPointCloud& demo_crop_pointcloud) {
                        return demo_crop_pointcloud.GetPathCroppedJSON();
                    },
                    "Path to saved selected polygon volume file.");
    docstring::ClassMethodDocInject(m, "DemoCropPointCloud", "path_pointcloud");
    docstring::ClassMethodDocInject(m, "DemoCropPointCloud",
                                    "path_cropped_json");
}

void pybind_demo_pointcloud_feature_matching(py::module& m) {
    // open3d.data.DemoPointCloudFeatureMatching
    py::class_<DemoPointCloudFeatureMatching,
               PySimpleDataset<DemoPointCloudFeatureMatching>,
               std::shared_ptr<DemoPointCloudFeatureMatching>, SimpleDataset>
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
                    "paths_pointclouds",
                    [](const DemoPointCloudFeatureMatching&
                               demo_feature_matching) {
                        return demo_feature_matching.GetPathsPointClouds();
                    },
                    "List of paths to point clouds, of size 2.")
            .def_property_readonly(
                    "paths_fpfh_features",
                    [](const DemoPointCloudFeatureMatching&
                               demo_feature_matching) {
                        return demo_feature_matching.GetPathsFPFHFeatures();
                    },
                    "List of paths to saved FPFH features binary for point "
                    "clouds, respectively, of size 2.")
            .def_property_readonly(
                    "paths_l32d_features",
                    [](const DemoPointCloudFeatureMatching&
                               demo_feature_matching) {
                        return demo_feature_matching.GetPathsL32DFeatures();
                    },
                    "List of paths to saved L32D features binary for point "
                    "clouds, respectively, of size 2.");
    docstring::ClassMethodDocInject(m, "DemoPointCloudFeatureMatching",
                                    "paths_pointclouds");
    docstring::ClassMethodDocInject(m, "DemoPointCloudFeatureMatching",
                                    "paths_fpfh_features");
    docstring::ClassMethodDocInject(m, "DemoPointCloudFeatureMatching",
                                    "paths_l32d_features");
}

void pybind_demo_pose_graph_optimization(py::module& m) {
    // open3d.data.DemoPoseGraphOptimization
    py::class_<DemoPoseGraphOptimization,
               PySimpleDataset<DemoPoseGraphOptimization>,
               std::shared_ptr<DemoPoseGraphOptimization>, SimpleDataset>
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
                    "path_pose_graph_fragment",
                    [](const DemoPoseGraphOptimization&
                               demo_pose_graph_optimization) {
                        return demo_pose_graph_optimization
                                .GetPathPoseGraphFragment();
                    },
                    "Path to example global pose graph (json).")
            .def_property_readonly(
                    "path_pose_graph_global",
                    [](const DemoPoseGraphOptimization&
                               demo_pose_graph_optimization) {
                        return demo_pose_graph_optimization
                                .GetPathPoseGraphGlobal();
                    },
                    "Path to example fragment pose graph (json).");
    docstring::ClassMethodDocInject(m, "DemoPoseGraphOptimization",
                                    "path_pose_graph_fragment");
    docstring::ClassMethodDocInject(m, "DemoPoseGraphOptimization",
                                    "path_pose_graph_global");
}

void pybind_sample_pointcloud_pcd(py::module& m) {
    // open3d.data.SamplePointCloudPCD
    py::class_<SamplePointCloudPCD, PySimpleDataset<SamplePointCloudPCD>,
               std::shared_ptr<SamplePointCloudPCD>, SimpleDataset>
            pcd_pointcloud(m, "SamplePointCloudPCD",
                           "Data class for `SamplePointCloudPCD` contains the "
                           "`fragment.pcd` point cloud mesh from the `Redwood "
                           "Living Room` dataset.");
    pcd_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SamplePointCloudPCD", "data_root"_a = "")
            .def_property_readonly(
                    "path",
                    [](const SamplePointCloudPCD& pcd_pointcloud) {
                        return pcd_pointcloud.GetPath();
                    },
                    "Path to the `pcd` format point cloud.");
    docstring::ClassMethodDocInject(m, "SamplePointCloudPCD", "path");
}

void pybind_sample_pointcloud_ply(py::module& m) {
    // open3d.data.SamplePointCloudPLY
    py::class_<SamplePointCloudPLY, PySimpleDataset<SamplePointCloudPLY>,
               std::shared_ptr<SamplePointCloudPLY>, SimpleDataset>
            ply_pointcloud(m, "SamplePointCloudPLY",
                           "Data class for `SamplePointCloudPLY` contains the "
                           "`fragment.pcd` point cloud mesh from the `Redwood "
                           "Living Room` dataset.");
    ply_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "SamplePointCloudPLY", "data_root"_a = "")
            .def_property_readonly(
                    "path",
                    [](const SamplePointCloudPLY& ply_pointcloud) {
                        return ply_pointcloud.GetPath();
                    },
                    "Path to the `ply` format point cloud.");
    docstring::ClassMethodDocInject(m, "SamplePointCloudPLY", "path");
}

void pybind_eagle(py::module& m) {
    // open3d.data.Eagle
    py::class_<Eagle, PySimpleDataset<Eagle>, std::shared_ptr<Eagle>,
               SimpleDataset>
            eagle(m, "Eagle",
                  "Data class for `Eagle` contains the `EaglePointCloud.ply` "
                  "file.");
    eagle.def(py::init<const std::string&, const std::string&>(),
              "prefix"_a = "Eagle", "data_root"_a = "")
            .def_property_readonly(
                    "path", [](const Eagle& eagle) { return eagle.GetPath(); },
                    "Path to the `EaglePointCloud.ply` file.");
    docstring::ClassMethodDocInject(m, "Eagle", "path");
}

void pybind_armadillo(py::module& m) {
    // open3d.data.Armadillo
    py::class_<Armadillo, PySimpleDataset<Armadillo>,
               std::shared_ptr<Armadillo>, SimpleDataset>
            armadillo(m, "Armadillo",
                      "Data class for `Armadillo` contains the "
                      "`ArmadilloMesh.ply` from the `Stanford 3D Scanning "
                      "Repository`.");
    armadillo
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "Armadillo", "data_root"_a = "")
            .def_property_readonly(
                    "path",
                    [](const Armadillo& armadillo) {
                        return armadillo.GetPath();
                    },
                    "Path to the `ArmadilloMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "Armadillo", "path");
}

void pybind_bunny(py::module& m) {
    // open3d.data.Bunny
    py::class_<Bunny, PySimpleDataset<Bunny>, std::shared_ptr<Bunny>,
               SimpleDataset>
            bunny(m, "Bunny",
                  "Data class for `Bunny` contains the `BunnyMesh.ply` from "
                  "the `Stanford 3D Scanning Repository`.");
    bunny.def(py::init<const std::string&, const std::string&>(),
              "prefix"_a = "Bunny", "data_root"_a = "")
            .def_property_readonly(
                    "path", [](const Bunny& bunny) { return bunny.GetPath(); },
                    "Path to the `BunnyMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "Bunny", "path");
}

void pybind_knot(py::module& m) {
    // open3d.data.Knot
    py::class_<Bunny, PySimpleDataset<Knot>, std::shared_ptr<Knot>,
               SimpleDataset>
            knot(m, "Knot",
                 "Data class for `Knot` contains the `KnotMesh.ply`.");
    knot.def(py::init<const std::string&, const std::string&>(),
             "prefix"_a = "Knot", "data_root"_a = "")
            .def_property_readonly(
                    "path", [](const Knot& knot) { return knot.GetPath(); },
                    "Path to the `KnotMesh.ply` file.");
    docstring::ClassMethodDocInject(m, "Knot", "path");
}

void pybind_juneau(py::module& m) {
    // open3d.data.Juneau
    py::class_<Bunny, PySimpleDataset<Juneau>, std::shared_ptr<Juneau>,
               SimpleDataset>
            juneau(m, "Juneau",
                   "Data class for `Juneau` contains the `JuneauImage.jpg` "
                   "file.");
    juneau.def(py::init<const std::string&, const std::string&>(),
               "prefix"_a = "Juneau", "data_root"_a = "")
            .def_property_readonly(
                    "path",
                    [](const Juneau& juneau) { return juneau.GetPath(); },
                    "Path to the `JuneauImage.jgp` file.");
    docstring::ClassMethodDocInject(m, "Juneau", "path");
}

void pybind_redwood_living_room_pointcloud(py::module& m) {
    py::class_<RedwoodLivingRoomPointClouds,
               PySimpleDataset<RedwoodLivingRoomPointClouds>,
               std::shared_ptr<RedwoodLivingRoomPointClouds>, SimpleDataset>
            redwood_living_room_pointcloud(
                    m, "RedwoodLivingRoomPointClouds",
                    "Dataset class for `RedwoodLivingRoomPointClouds` contains "
                    "57 `pointclouds` of `ply binary` format.");
    redwood_living_room_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "RedwoodLivingRoomPointClouds",
                 "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const RedwoodLivingRoomPointClouds&
                               redwood_living_room_pointcloud) {
                        return redwood_living_room_pointcloud.GetPaths();
                    },
                    "List of paths to ply point-cloud fragments of size 56.");
    docstring::ClassMethodDocInject(m, "RedwoodLivingRoomPointClouds", "paths");
}

void pybind_redwood_office_pointcloud(py::module& m) {
    py::class_<RedwoodOfficePointClouds,
               PySimpleDataset<RedwoodOfficePointClouds>,
               std::shared_ptr<RedwoodOfficePointClouds>, SimpleDataset>
            redwood_office_pointcloud(
                    m, "RedwoodOfficePointClouds",
                    "Dataset class for `RedwoodOfficePointClouds` contains 53 "
                    "`pointclouds` of `ply binary` format.");
    redwood_office_pointcloud
            .def(py::init<const std::string&, const std::string&>(),
                 "prefix"_a = "RedwoodOfficePointClouds", "data_root"_a = "")
            .def_property_readonly(
                    "paths",
                    [](const RedwoodOfficePointClouds&
                               redwood_office_pointcloud) {
                        return redwood_office_pointcloud.GetPaths();
                    },
                    "List of paths to ply point-cloud fragments of size 53.");
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
    pybind_eagle(m_submodule);
    pybind_armadillo(m_submodule);
    pybind_bunny(m_submodule);
    pybind_knot(m_submodule);
    pybind_juneau(m_submodule);
    pybind_redwood_living_room_pointcloud(m_submodule);
    pybind_redwood_office_pointcloud(m_submodule);
}

}  // namespace data
}  // namespace open3d
