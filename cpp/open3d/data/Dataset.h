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
    std::string GetPathPointCloud() const { return path_pointcloud_; }
    /// \brief Returns path to saved selected polygon volume file.
    std::string GetPathCroppedJSON() const { return path_cropped_json_; }

private:
    // Path to example point cloud.
    std::string path_pointcloud_;
    // Path to saved selected polygon volume file.
    std::string path_cropped_json_;
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

    /// \brief Returns list of paths to point clouds, of size 2.
    std::vector<std::string> GetPathsPointClouds() const {
        return paths_pointclouds_;
    }
    /// \brief Returns list of paths to saved FPFH features binary for point
    /// clouds, respectively, of size 2.
    std::vector<std::string> GetPathsFPFHFeatures() const {
        return paths_fpfh_features_;
    }
    /// \brief Returns list of paths to saved L32D features binary for point
    /// clouds, respectively, of size 2.
    std::vector<std::string> GetPathsL32DFeatures() const {
        return paths_l32d_features_;
    }

private:
    /// List of paths to point clouds, of size 2.
    std::vector<std::string> paths_pointclouds_;
    /// List of saved FPFH features binary for point clouds,
    /// respectively, of size 2.
    std::vector<std::string> paths_fpfh_features_;
    /// List of saved L32D features binary for point clouds,
    /// respectively, of size 2.
    std::vector<std::string> paths_l32d_features_;
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
    std::string GetPathPoseGraphFragment() const {
        return path_pose_graph_fragment_;
    }
    /// \brief Returns path to example fragment pose graph (json).
    std::string GetPathPoseGraphGlobal() const {
        return path_pose_graph_global_;
    }

private:
    /// Path to example global pose graph (json).
    std::string path_pose_graph_fragment_;
    /// Path to example fragment pose graph (json).
    std::string path_pose_graph_global_;
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
    std::string GetPathColor() const { return path_color_; };
    /// \brief Returns path to depth image sample.
    std::string GetPathDepth() const { return path_depth_; };

private:
    /// Path to color image sample.
    std::string path_color_;
    /// Path to depth image sample.
    std::string path_depth_;
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
    std::string GetPathColor() const { return path_color_; };
    /// \brief Returns path to depth image sample.
    std::string GetPathDepth() const { return path_depth_; };

private:
    /// Path to color image sample.
    std::string path_color_;
    /// Path to depth image sample.
    std::string path_depth_;
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
    std::string GetPathColor() const { return path_color_; };
    /// \brief Returns path to depth image sample.
    std::string GetPathDepth() const { return path_depth_; };

private:
    /// Path to color image sample.
    std::string path_color_;
    /// Path to depth image sample.
    std::string path_depth_;
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

    /// \brief Returns List of paths to color image samples of size 5.
    std::vector<std::string> GetPathsColor() const { return paths_color_; };
    /// \brief Returns List of paths to depth image samples of size 5.
    std::vector<std::string> GetPathsDepth() const { return paths_depth_; };

    /// \brief Returns path to camera trajectory log file `trajectory.log`.
    std::string GetPathTrajectoryLog() const { return path_trajectory_log_; };
    /// \brief Returns path to camera trajectory log file `trajectory.log`.
    std::string GetPathOdometryLog() const { return path_odometry_log_; };
    /// \brief Returns path to color and depth image match file `rgbd.match`.
    std::string GetPathRGBDMatch() const { return path_rgbd_match_; };
    /// \brief Returns path to pointcloud reconstruction from TSDF.
    std::string GetPathReconstruction() const { return path_reconstruction_; };

private:
    /// List of paths to color image samples of size 5.
    std::vector<std::string> paths_color_;
    /// List of paths to depth image samples of size 5.
    std::vector<std::string> paths_depth_;

    /// Path to camera trajectory log file `trajectory.log`.
    std::string path_trajectory_log_;
    /// Path to camera odometry log file `odometry.log`.
    std::string path_odometry_log_;
    /// Path to color and depth image match file `rgbd.match`.
    std::string path_rgbd_match_;
    /// Path to pointcloud reconstruction from TSDF.
    std::string path_reconstruction_;
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

    std::vector<std::string> GetPathsColor() const { return paths_color_; };
    std::vector<std::string> GetPathsDepth() const { return paths_depth_; };
    std::string GetPathKeyframePosesLog() const {
        return path_keyframe_poses_log_;
    };
    std::string GetPathReconstruction() const { return path_reconstruction_; };

private:
    std::vector<std::string> paths_color_;
    std::vector<std::string> paths_depth_;
    std::string path_keyframe_poses_log_;
    std::string path_reconstruction_;
};

/// \class Eagle
/// \brief Data class for `Eagle` contains the `EaglePointCloud.ply` file.
class Eagle : public SimpleDataset {
public:
    Eagle(const std::string& prefix = "Eagle",
          const std::string& data_root = "");

    /// \brief Returns path to the `EaglePointCloud.ply`.
    std::string GetPath() const { return path_; };

private:
    /// Path to `EaglePointCloud.ply` file.
    std::string path_;
};

/// \class Armadillo
/// \brief Data class for `Armadillo` contains the `ArmadilloMesh.ply` from the
/// `Stanford 3D Scanning Repository`.
class Armadillo : public SimpleDataset {
public:
    Armadillo(const std::string& prefix = "Armadillo",
              const std::string& data_root = "");

    /// \brief Returns path to the `ArmadilloMesh.ply`.
    std::string GetPath() const { return path_; };

private:
    /// Path to the `ArmadilloMesh.ply` file.
    std::string path_;
};

/// \class Bunny
/// \brief Data class for `Bunny` contains the `BunnyMesh.ply` from the
/// `Stanford 3D Scanning Repository`.
class Bunny : public SimpleDataset {
public:
    Bunny(const std::string& prefix = "Bunny",
          const std::string& data_root = "");

    /// \brief Returns path to the `BunnyMesh.ply`.
    std::string GetPath() const { return path_; };

private:
    /// Path to `BunnyMesh.ply` file.
    std::string path_;
};

/// \class Knot
/// \brief Data class for `Knot` contains the `KnotMesh.ply` file.
class Knot : public SimpleDataset {
public:
    Knot(const std::string& prefix = "Knot", const std::string& data_root = "");

    /// \brief Returns path to the `KnotMesh.ply`.
    std::string GetPath() const { return path_; };

private:
    /// Path to `KnotMesh.ply` file.
    std::string path_;
};

/// \class Juneau
/// \brief Data class for `Juneau` contains the `JuneauImage.jpg` file.
class Juneau : public SimpleDataset {
public:
    Juneau(const std::string& prefix = "Juneau",
           const std::string& data_root = "");

    /// \brief Returns path to the `JuneauImage.jgp`.
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
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the ply point-cloud fragment at index (from 0 to
    /// 57).
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
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the ply point-cloud fragment at index (from 0 to
    /// 53).
    std::string GetPaths(size_t index) const;

private:
    /// List of paths to ply point-cloud fragments of size 53.
    std::vector<std::string> paths_;
};

}  // namespace data
}  // namespace open3d
