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
/// \brief Dataset class for `DemoICPPointClouds` contains 3 `pointclouds` of
/// `pcd binary` format. These pointclouds have `positions, colors, normals,
/// curvatures`. This dataset is used in Open3D for ICP demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoICPPointClouds : public SimpleDataset {
public:
    DemoICPPointClouds(const std::string& prefix = "DemoICPPointClouds",
                       const std::string& data_root = "");

    /// \brief Returns list of list of 3 point cloud paths.
    /// Use `GetPaths()[0]`, `GetPaths()[1]`, and `GetPaths()[2]` to access the
    /// paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the point cloud at index.
    /// Use `GetPaths(0)`, `GetPaths(1)`, and `GetPaths(2)` to access the paths.
    std::string GetPaths(size_t index) const;

private:
    // List of path to PCD point-cloud fragments.
    // paths_[x] returns path to `cloud_bin_x.pcd` where x is from
    // 0 to 2.
    std::vector<std::string> paths_;
};

/// \class DemoColoredICPPointClouds
/// \brief Dataset class for `DemoICPPointClouds` contains 3 `pointclouds` of
/// `pcd binary` format. These pointclouds have `positions, colors, normals,
/// curvatures`. This dataset is used in Open3D for ICP demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoColoredICPPointClouds : public SimpleDataset {
public:
    DemoColoredICPPointClouds(
            const std::string& prefix = "DemoColoredICPPointClouds",
            const std::string& data_root = "");

    /// \brief Returns list of list of 2 point cloud paths.
    /// Use `GetPaths()[0]`, and `GetPaths()[1]` to access the paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the point cloud at index.
    /// Use `GetPaths(0)`, and `GetPaths(1)` to access the paths.
    std::string GetPaths(size_t index) const;

private:
    // List of path to PCD point-cloud fragments.
    // paths_[x] returns path to `cloud_bin_x.pcd` where x is from
    // 0 to 2.
    std::vector<std::string> paths_;
};

class DemoCropPointCloud : public SimpleDataset {
public:
    DemoCropPointCloud(const std::string& prefix = "DemoCropPointCloud",
                       const std::string& data_root = "");

    std::string GetPathPointCloud() const { return path_pointcloud_; }
    std::string GetPathCroppedJSON() const { return path_cropped_json_; }

private:
    std::string path_pointcloud_;
    std::string path_cropped_json_;
};

class DemoPointCloudFeatureMatching : public SimpleDataset {
public:
    DemoPointCloudFeatureMatching(
            const std::string& prefix = "DemoPointCloudFeatureMatching",
            const std::string& data_root = "");

    std::vector<std::string> GetPathsPointClouds() const {
        return paths_pointclouds_;
    }
    std::vector<std::string> GetPathsFPFHFeatures() const {
        return paths_fpfh_features_;
    }
    std::vector<std::string> GetPathsL32DFeatures() const {
        return paths_l32d_features_;
    }

private:
    std::vector<std::string> paths_pointclouds_;
    std::vector<std::string> paths_fpfh_features_;
    std::vector<std::string> paths_l32d_features_;
};

class DemoPoseGraphOptimization : public SimpleDataset {
public:
    DemoPoseGraphOptimization(
            const std::string& prefix = "DemoPoseGraphOptimization",
            const std::string& data_root = "");

    std::string GetPathPoseGraphFragment() const {
        return path_pose_graph_fragment_;
    }
    std::string GetPathPoseGraphGlobal() const {
        return path_pose_graph_global_;
    }

private:
    std::string path_pose_graph_fragment_;
    std::string path_pose_graph_global_;
};

class Armadillo : public SimpleDataset {
public:
    Armadillo(const std::string& prefix = "Armadillo",
              const std::string& data_root = "");

    /// \brief Returns path to the bunny.ply pointcloud.
    std::string GetPath() const { return path_; };

private:
    // path to Armadillo.ply file.
    std::string path_;
};

class Bunny : public SimpleDataset {
public:
    Bunny(const std::string& prefix = "Bunny",
          const std::string& data_root = "");

    /// \brief Returns path to the bunny.ply pointcloud.
    std::string GetPath() const { return path_; };

private:
    // path to Bunny.ply file.
    std::string path_;
};

/// \class RedwoodLivingRoomFragments
/// \brief Dataset class for `RedwoodLivingRoomFragments` contains 57
/// `pointclouds` of `ply binary` format. These pointclouds have positions,
/// colors, normals, curvatures.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class RedwoodLivingRoomFragments : public SimpleDataset {
public:
    RedwoodLivingRoomFragments(
            const std::string& prefix = "RedwoodLivingRoomFragments",
            const std::string& data_root = "");

    /// \brief GetPaths()[x] returns path to `cloud_bin_x.ply` pointcloud, where
    /// x is between 0 to 56.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the pointcloud at index.
    /// GetPaths(x) returns path to `cloud_bin_x.ply` pointcloud, where x is
    /// between 0 to 56.
    std::string GetPaths(size_t index) const;

private:
    // Path to PLY point-cloud fragments.
    // paths_[x] return path to `cloud_bin_x.ply` where x is from
    // 0 to 56.
    std::vector<std::string> paths_;
};

/// \class RedwoodOfficeFragments
/// \brief Dataset class for `RedwoodOfficeFragments` contains 51
/// `pointclouds` of `ply binary` format. These pointclouds have positions,
/// colors, normals, curvatures.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class RedwoodOfficeFragments : public SimpleDataset {
public:
    RedwoodOfficeFragments(const std::string& prefix = "RedwoodOfficeFragments",
                           const std::string& data_root = "");

    /// \brief GetPaths()[x] returns path to `cloud_bin_x.ply` pointcloud, where
    /// X is between 0 to 51.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Returns path to the pointcloud at index.
    /// GetPaths(x) returns path to `cloud_bin_x.ply` pointcloud, where x is
    /// between 0 to 51.
    std::string GetPaths(size_t index) const;

private:
    // Path to PLY point-cloud fragments.
    // paths_[x] return path to `cloud_bin_x.ply` where x is from
    // 0 to 51.
    std::vector<std::string> paths_;
};

}  // namespace data
}  // namespace open3d
