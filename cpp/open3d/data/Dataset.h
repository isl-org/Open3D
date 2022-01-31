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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

/// Function to return default data root directory in the following order:
///
/// (a) OPEN3D_DATA_ROOT environment variable.
/// (b) $HOME/open3d_data.
std::string LocateDataRoot();

/// \class Dataset
/// \brief Base Open3D dataset class.
///
/// The Dataset classes in Open3D are designed for convenient access to
/// "built-in" example and test data. You'll need internet access to use the
/// dataset classes. The downloaded data will be stored in the Open3D's data
/// root directory.
///
/// - A dataset class locates the data root directory in the following order:
///   (a) User-specified by `data_root` when instantiating a dataset object.
///   (b) OPEN3D_DATA_ROOT environment variable.
///   (c) $HOME/open3d_data.
///   By default, (c) will be used, and it is also the recommended way.
/// - When a dataset object is instantiated, the corresponding data will be
///   downloaded in `${data_root}/download/prefix/` and extracted or copied to
///   `${data_root}/extract/prefix/`. If the extracted data directory exists,
///   the files will be used without validation. If it does not exists, and the
///   valid downloaded file exists, the data will be extracted from the
///   downloaded file. If downloaded file does not exists, or validates against
///   the provided MD5, it will be re-downloaded.
/// - After the data is downloaded and extracted, the dataset object will NOT
///   load the data for you. Instead, you will get the paths to the data files
///   and use Open3D's I/O functions to load the data. This design exposes where
///   the data is stored and how the data is loaded, allowing users to modify
///   the code and load their own data in a similar way. Please check the
///   documentation of the specific dataset to know more about the specific
///   functionalities provided for it.
class Dataset {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param prefix Prefix of the dataset. The data is downloaded in
    /// `${data_root}/download/${prefix}/` and extracted in
    /// `${data_root}/extract/${prefix}/`.
    /// \param data_root Path to `${data_root}`, which contains all the
    /// downloaded and extracted files.
    /// The data root directory is located in the following order:
    ///   (a) User-specified by `data_root` when instantiating a dataset object.
    ///   (b) OPEN3D_DATA_ROOT environment variable.
    ///   (c) $HOME/open3d_data.
    ///   By default, (c) will be used, and it is also the recommended way.
    Dataset(const std::string& prefix, const std::string& data_root = "");

    virtual ~Dataset() {}

    /// \brief Get data root directory. The data root is set at construction
    /// time or automatically determined.
    const std::string GetDataRoot() const { return data_root_; }
    /// \brief Get prefix for the dataset.
    const std::string GetPrefix() const { return prefix_; }

    /// \brief Get absolute path to download directory. i.e.
    /// ${data_root}/${download_prefix}/${prefix}
    const std::string GetDownloadDir() const {
        return GetDataRoot() + "/download/" + GetPrefix();
    }
    /// \brief Get absolute path to extract directory. i.e.
    /// ${data_root}/${extract_prefix}/${prefix}
    const std::string GetExtractDir() const {
        return GetDataRoot() + "/extract/" + GetPrefix();
    }

protected:
    /// Open3D data root.
    std::string data_root_;
    /// Dataset prefix.
    std::string prefix_;
};

/// \class SimpleDataset
/// \brief This class allows user to create simple dataset which includes single
/// file downloading and extracting / copying.
class SimpleDataset : public Dataset {
public:
    SimpleDataset(const std::string& prefix,
                  const std::vector<std::string>& urls,
                  const std::string& md5,
                  const bool no_extract = false,
                  const std::string& data_root = "");

    virtual ~SimpleDataset() {}
};

/// \class DemoICPPointClouds
/// \brief Data class for `DemoICPPointClouds` contains 3 `pointclouds` of
/// `pcd binary` format. This data is used in Open3D for ICP demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoICPPointClouds : public SimpleDataset {
public:
    DemoICPPointClouds(const std::string& prefix = "DemoICPPointClouds",
                       const std::string& data_root = "");

    /// \brief Returns list of 3 point cloud paths. Use `GetPaths()[0]`,
    /// `GetPaths()[1]`, and `GetPaths()[2]` to access the paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the point cloud at index. Use `GetPaths(0)`,
    /// `GetPaths(1)`, and `GetPaths(2)` to access the paths.
    std::string GetPaths(size_t index) const;

private:
    // List of path to PCD point-cloud fragments.
    std::vector<std::string> paths_;
};

/// \class DemoColoredICPPointClouds
/// \brief Data class for `DemoColoredICPPointClouds` contains 2
/// `pointclouds` of `ply` format. This data is used in Open3D for
/// Colored-ICP demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoColoredICPPointClouds : public SimpleDataset {
public:
    DemoColoredICPPointClouds(
            const std::string& prefix = "DemoColoredICPPointClouds",
            const std::string& data_root = "");

    /// \brief Returns list of list of 2 point cloud paths. Use `GetPaths()[0]`,
    /// and `GetPaths()[1]` to access the paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the point cloud at index. Use `GetPaths(0)`, and
    /// `GetPaths(1)` to access the paths.
    std::string GetPaths(size_t index) const;

private:
    // List of path to PCD point-cloud fragments.
    std::vector<std::string> paths_;
};

/// \class DemoCropPointCloud
/// \brief Data class for `DemoCropPointCloud` contains a point cloud, and
/// `cropped.json` (a saved selected polygon volume file). This data is used
/// in Open3D for point cloud crop demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoCropPointCloud : public SimpleDataset {
public:
    DemoCropPointCloud(const std::string& prefix = "DemoCropPointCloud",
                       const std::string& data_root = "");

    /// \brief Returns path to example point cloud.
    std::string GetPointCloudPath() const { return pointcloud_path_; }
    /// \brief Returns path to saved selected polygon volume file.
    std::string GetCroppedJSONPath() const { return cropped_json_path_; }

private:
    // Path to example point cloud.
    std::string pointcloud_path_;
    // Path to saved selected polygon volume file.
    std::string cropped_json_path_;
};

/// \class DemoPointCloudFeatureMatching
/// \brief Data class for `DemoPointCloudFeatureMatching` contains 2
/// pointcloud fragments and their respective FPFH features and L32D features.
/// This data is used in Open3D for point cloud feature matching demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoPointCloudFeatureMatching : public SimpleDataset {
public:
    DemoPointCloudFeatureMatching(
            const std::string& prefix = "DemoPointCloudFeatureMatching",
            const std::string& data_root = "");

    /// \brief Returns list of paths to point clouds, of size 2. Use
    /// `GetPointCloudPaths()[0]`, and `GetPointCloudPaths()[1]` to access the
    /// paths.
    std::vector<std::string> GetPointCloudPaths() const {
        return pointcloud_paths_;
    }
    /// \brief Returns list of paths to saved FPFH features binary for point
    /// clouds, respectively, of size 2. Use `GetFPFHFeaturePaths()[0]`, and
    /// `GetFPFHFeaturePaths()[1]` to access the paths.
    std::vector<std::string> GetFPFHFeaturePaths() const {
        return fpfh_feature_paths_;
    }
    /// \brief Returns list of paths to saved L32D features binary for point
    /// clouds, respectively, of size 2. Use `GetL32DFeaturePaths()[0]`, and
    /// `GetL32DFeaturePaths()[1]` to access the paths.
    std::vector<std::string> GetL32DFeaturePaths() const {
        return l32d_feature_paths_;
    }

private:
    /// List of paths to point clouds, of size 2.
    std::vector<std::string> pointcloud_paths_;
    /// List of saved FPFH features binary for point clouds,
    /// respectively, of size 2.
    std::vector<std::string> fpfh_feature_paths_;
    /// List of saved L32D features binary for point clouds,
    /// respectively, of size 2.
    std::vector<std::string> l32d_feature_paths_;
};

/// \class DemoPoseGraphOptimization
/// \brief Data class for `DemoPoseGraphOptimization` contains an example
/// fragment pose graph, and global pose graph. This data is used in Open3D
/// for pose graph optimization demo.
class DemoPoseGraphOptimization : public SimpleDataset {
public:
    DemoPoseGraphOptimization(
            const std::string& prefix = "DemoPoseGraphOptimization",
            const std::string& data_root = "");

    /// \brief Returns path to example global pose graph (json).
    std::string GetPoseGraphFragmentPath() const {
        return pose_graph_fragment_path_;
    }
    /// \brief Returns path to example fragment pose graph (json).
    std::string GetPoseGraphGlobalPath() const {
        return pose_graph_global_path_;
    }

private:
    /// Path to example global pose graph (json).
    std::string pose_graph_fragment_path_;
    /// Path to example fragment pose graph (json).
    std::string pose_graph_global_path_;
};

/// \class SamplePointCloudPCD
/// \brief Data class for `SamplePointCloudPCD` contains the `fragment.pcd`
/// point cloud mesh from the `Redwood Living Room` dataset.
class SamplePointCloudPCD : public SimpleDataset {
public:
    SamplePointCloudPCD(const std::string& prefix = "SamplePointCloudPCD",
                        const std::string& data_root = "");

    /// \brief Returns path to the `pcd` format point cloud.
    std::string GetPath() const { return path_; };

private:
    /// Path to the `pcd` format point cloud.
    std::string path_;
};

/// \class SamplePointCloudPLY
/// \brief Data class for `SamplePointCloudPLY` contains the `fragment.ply`
/// point cloud mesh from the `Redwood Living Room` dataset.
class SamplePointCloudPLY : public SimpleDataset {
public:
    SamplePointCloudPLY(const std::string& prefix = "SamplePointCloudPLY",
                        const std::string& data_root = "");

    /// \brief Returns path to the `ply` format point cloud.
    std::string GetPath() const { return path_; };

private:
    /// Path to the `ply` format point cloud.
    std::string path_;
};

/// \class SampleRGBDImageNYU
/// \brief Data class for `SampleRGBDImageNYU` contains a color image
/// `NYU_color.ppm` and a depth image `NYU_depth.pgm` sample from NYU RGBD
/// dataset.
class SampleRGBDImageNYU : public SimpleDataset {
public:
    SampleRGBDImageNYU(const std::string& prefix = "SampleRGBDImageNYU",
                       const std::string& data_root = "");

    /// \brief Returns path to color image sample.
    std::string GetColorPath() const { return color_path_; };
    /// \brief Returns path to depth image sample.
    std::string GetDepthPath() const { return depth_path_; };

private:
    /// Path to color image sample.
    std::string color_path_;
    /// Path to depth image sample.
    std::string depth_path_;
};

/// \class SampleRGBDImageSUN
/// \brief Data class for `SampleRGBDImageSUN` contains a color image
/// `SUN_color.jpg` and a depth image `SUN_depth.png` sample from SUN RGBD
/// dataset.
class SampleRGBDImageSUN : public SimpleDataset {
public:
    SampleRGBDImageSUN(const std::string& prefix = "SampleRGBDImageSUN",
                       const std::string& data_root = "");

    /// \brief Returns path to color image sample.
    std::string GetColorPath() const { return color_path_; };
    /// \brief Returns path to depth image sample.
    std::string GetDepthPath() const { return depth_path_; };

private:
    /// Path to color image sample.
    std::string color_path_;
    /// Path to depth image sample.
    std::string depth_path_;
};

/// \class SampleRGBDImageTUM
/// \brief Data class for `SampleRGBDImageTUM` contains a color image
/// `TUM_color.png` and a depth image `TUM_depth.png` sample from TUM RGBD
/// dataset.
class SampleRGBDImageTUM : public SimpleDataset {
public:
    SampleRGBDImageTUM(const std::string& prefix = "SampleRGBDImageTUM",
                       const std::string& data_root = "");

    /// \brief Returns path to color image sample.
    std::string GetColorPath() const { return color_path_; };
    /// \brief Returns path to depth image sample.
    std::string GetDepthPath() const { return depth_path_; };

private:
    /// Path to color image sample.
    std::string color_path_;
    /// Path to depth image sample.
    std::string depth_path_;
};

/// \class SampleRGBDDatasetICL
/// \brief Data class for `SampleRGBDDatasetICL` contains a sample set of 5
/// color and depth images from ICL-NUIM RGBD dataset living-room1. It also
/// contains example `camera trajectory log`, `odometry log`, `rgbd match`, and
/// `point cloud reconstruction using TSDF`.
class SampleRGBDDatasetICL : public SimpleDataset {
public:
    SampleRGBDDatasetICL(const std::string& prefix = "SampleRGBDDatasetICL",
                         const std::string& data_root = "");

    /// \brief Returns List of paths to color image samples of size 5. Use
    /// `GetColorPaths()[0]`, `GetColorPaths()[1]` ... `GetColorPaths()[4]` to
    /// access the paths.
    std::vector<std::string> GetColorPaths() const { return color_paths_; };
    /// \brief Returns List of paths to depth image samples of size 5. Use
    /// `GetDepthPaths()[0]`, `GetDepthPaths()[1]` ... `GetDepthPaths()[4]` to
    /// access the paths.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; };

    /// \brief Returns path to camera trajectory log file `trajectory.log`.
    std::string GetTrajectoryLogPath() const { return trajectory_log_path_; };
    /// \brief Returns path to camera trajectory log file `trajectory.log`.
    std::string GetOdometryLogPath() const { return odometry_log_path_; };
    /// \brief Returns path to color and depth image match file `rgbd.match`.
    std::string GetRGBDMatchPath() const { return rgbd_match_path_; };
    /// \brief Returns path to pointcloud reconstruction from TSDF.
    std::string GetReconstructionPath() const { return reconstruction_path_; };

private:
    /// List of paths to color image samples of size 5.
    std::vector<std::string> color_paths_;
    /// List of paths to depth image samples of size 5.
    std::vector<std::string> depth_paths_;

    /// Path to camera trajectory log file `trajectory.log`.
    std::string trajectory_log_path_;
    /// Path to camera odometry log file `odometry.log`.
    std::string odometry_log_path_;
    /// Path to color and depth image match file `rgbd.match`.
    std::string rgbd_match_path_;
    /// Path to pointcloud reconstruction from TSDF.
    std::string reconstruction_path_;
};

/// \class SampleFountainRGBDDataset
/// \brief Data class for `SampleFountainRGBDDataset` contains a sample set of
/// 33 color and depth images from the `Fountain RGBD dataset`. It also
/// contains `camera poses at keyframes log` and `mesh reconstruction`. It is
/// used in demo of `Color Map Optimization`.
class SampleFountainRGBDDataset : public SimpleDataset {
public:
    SampleFountainRGBDDataset(
            const std::string& prefix = "SampleFountainRGBDDataset",
            const std::string& data_root = "");

    /// \brief Returns List of paths to color image samples of size 33. Use
    /// `GetColorPaths()[0]`, `GetColorPaths()[1]` ... `GetColorPaths()[32]` to
    /// access the paths.
    std::vector<std::string> GetColorPaths() const { return color_paths_; };
    /// \brief Returns List of paths to depth image samples of size 5. Use
    /// `GetDepthPaths()[0]`, `GetDepthPaths()[1]` ... `GetDepthPaths()[4]` to
    /// access the paths.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; };
    /// \brief Returns path to camera poses at keyfragmes log file `key.log`.
    std::string GetKeyframePosesLogPath() const {
        return keyframe_poses_log_path_;
    };
    /// \brief Returns path to mesh reconstruction.
    std::string GetReconstructionPath() const { return reconstruction_path_; };

private:
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::string keyframe_poses_log_path_;
    std::string reconstruction_path_;
};

/// \class EaglePointCloud
/// \brief Data class for `EaglePointCloud` contains the `EaglePointCloud.ply`
/// file.
class EaglePointCloud : public SimpleDataset {
public:
    EaglePointCloud(const std::string& prefix = "EaglePointCloud",
                    const std::string& data_root = "");

    /// \brief Returns path to the `EaglePointCloud.ply` file.
    std::string GetPath() const { return path_; };

private:
    /// Path to `EaglePointCloud.ply` file.
    std::string path_;
};

/// \class ArmadilloMesh
/// \brief Data class for `ArmadilloMesh` contains the `ArmadilloMesh.ply` from
/// the `Stanford 3D Scanning Repository`.
class ArmadilloMesh : public SimpleDataset {
public:
    ArmadilloMesh(const std::string& prefix = "ArmadilloMesh",
                  const std::string& data_root = "");

    /// \brief Returns path to the `ArmadilloMesh.ply` file.
    std::string GetPath() const { return path_; };

private:
    /// Path to the `ArmadilloMesh.ply` file.
    std::string path_;
};

/// \class BunnyMesh
/// \brief Data class for `BunnyMesh` contains the `BunnyMesh.ply` from the
/// `Stanford 3D Scanning Repository`.
class BunnyMesh : public SimpleDataset {
public:
    BunnyMesh(const std::string& prefix = "BunnyMesh",
              const std::string& data_root = "");

    /// \brief Returns path to the `BunnyMesh.ply` file.
    std::string GetPath() const { return path_; };

private:
    /// Path to `BunnyMesh.ply` file.
    std::string path_;
};

/// \class KnotMesh
/// \brief Data class for `KnotMesh` contains the `KnotMesh.ply` file.
class KnotMesh : public SimpleDataset {
public:
    KnotMesh(const std::string& prefix = "KnotMesh",
             const std::string& data_root = "");

    /// \brief Returns path to the `KnotMesh.ply` file.
    std::string GetPath() const { return path_; };

private:
    /// Path to `KnotMesh.ply` file.
    std::string path_;
};

/// \class JuneauImage
/// \brief Data class for `JuneauImage` contains the `JuneauImage.jpg` file.
class JuneauImage : public SimpleDataset {
public:
    JuneauImage(const std::string& prefix = "JuneauImage",
                const std::string& data_root = "");

    /// \brief Returns path to the `JuneauImage.jgp` file.
    std::string GetPath() const { return path_; };

private:
    /// Path to `JuneauImage.jgp` file.
    std::string path_;
};

/// \class RedwoodLivingRoomPointClouds
/// \brief Dataset class for `RedwoodLivingRoomPointClouds` contains 57
/// `pointclouds` of `ply binary` format.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class RedwoodLivingRoomPointClouds : public SimpleDataset {
public:
    RedwoodLivingRoomPointClouds(
            const std::string& prefix = "RedwoodLivingRoomPointClouds",
            const std::string& data_root = "");

    /// \brief Returns list of paths to ply point-cloud fragments of size 57.
    /// Use `GetPaths()[0]`, `GetPaths()[1]` ... `GetPaths()[56]` to access the
    /// paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the ply point-cloud fragment at index (from 0 to
    /// 57). Use `GetPaths(0)`, `GetPaths(1)` ... `GetPaths(56)` to access the
    /// paths.
    std::string GetPaths(size_t index) const;

private:
    /// List of paths to ply point-cloud fragments of size 57.
    std::vector<std::string> paths_;
};

/// \class RedwoodOfficePointClouds
/// \brief Dataset class for `RedwoodOfficePointClouds` contains 53
/// `pointclouds` of `ply binary` format.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class RedwoodOfficePointClouds : public SimpleDataset {
public:
    RedwoodOfficePointClouds(
            const std::string& prefix = "RedwoodOfficePointClouds",
            const std::string& data_root = "");

    /// \brief Returns list of paths to ply point-cloud fragments of size 53.
    /// Use `GetPaths()[0]`, `GetPaths()[1]` ... `GetPaths()[52]` to access the
    /// paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the ply point-cloud fragment at index (from 0 to
    /// 53). Use `GetPaths(0)`, `GetPaths(1)` ... `GetPaths(52)` to access the
    /// paths.
    std::string GetPaths(size_t index) const;

private:
    /// List of paths to ply point-cloud fragments of size 53.
    std::vector<std::string> paths_;
};

}  // namespace data
}  // namespace open3d
