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
///   downloaded and extracted. If the data already exists and the checksum
///   matches, the download will be skipped.
/// - After the data is downloaded and extracted, the dataset object will NOT
///   load the data for you. Instead, you will get the paths to the data files
///   and use Open3D's I/O functions to load the data. This design exposes where
///   the data is stored and how the data is loaded, allowing users to modify
///   the code and load their own data in a similar way.
class Dataset {
public:
    Dataset(const std::string& prefix,
            const std::string& help_string = "",
            const std::string& data_root = "");

    virtual ~Dataset() {}

    /// Get data root directory. The data root is set at construction time or
    /// automatically determined.
    const std::string GetDataRoot() const { return data_root_; }
    const std::string GetPrefix() const { return prefix_; }
    const std::string GetHelpString() const { return help_string_; }
    const std::string GetExtractDir(
            const bool relative_to_data_root = false) const {
        return relative_to_data_root ? "extract/" + prefix_
                                     : data_root_ + "/extract/" + prefix_;
    }
    const std::string GetDownloadDir(
            const bool relative_to_data_root = false) const {
        return relative_to_data_root ? "download/" + prefix_
                                     : data_root_ + "/download/" + prefix_;
    }

protected:
    /// Open3D data root.
    std::string data_root_;
    /// Dataset prefix.
    std::string prefix_;
    /// Dataset help string containing informations such as source,
    /// documentation link, functionalities, usage, licence, and other useful
    /// informations.
    std::string help_string_;
};

class SimpleDataset : public Dataset {
public:
    SimpleDataset(const std::string& prefix,
                  const std::vector<std::string>& urls,
                  const std::string& md5,
                  const bool no_extract = false,
                  const std::string& help_string = "",
                  const std::string& data_root = "");

    virtual ~SimpleDataset() {}
};

namespace dataset {

class SampleICPPointClouds : public SimpleDataset {
public:
    SampleICPPointClouds(const std::string& prefix = "SampleICPPointClouds",
                         const std::string& data_root = "")
        : SimpleDataset(
                  prefix,
                  {"https://github.com/isl-org/open3d_downloads/releases/"
                   "download/sample-icp-pointclouds/SampleICPPointClouds.zip"},
                  "3ee7a2631caa3c47a333972e3c4fb315") {
        for (int i = 0; i < 3; ++i) {
            path_to_pointclouds_.push_back(Dataset::GetExtractDir() +
                                           "/cloud_bin_" + std::to_string(i) +
                                           ".pcd");
        }

        Dataset::help_string_ = std::string(R""""(
Colored point-cloud fragments of living-room from ICL-NUIM
RGBD Benchmark Dataset in PCD format.

Information:
- Type: Point cloud fragments [contains points, colors, normals, curvature].
- Format: PCD Binary.
- Source: ICL-NUIM RGBD Benchmark Dataset.
- MD5: 4d39442a86e9fe80c967a6c513d57442

Contents of SampleICPPointClouds.zip:
    SampleICPPointClouds
    ├── cloud_bin_0.pcd
    ├── cloud_bin_1.pcd
    ├── cloud_bin_2.pcd
    └── init.log

Data Members:
    path_to_fragments_ : List of path to PCD point-cloud fragments.
                         path_to_fragments_[x] returns path to `cloud_bin_x.pcd`
                         where x is from 0 to 2.

Application: Used in Open3D ICP registration demo examples.

Licence: The data is released under Creative Commons 3.0 (CC BY 3.0),
         see http://creativecommons.org/licenses/by/3.0/.
     )"""");
    }

    std::vector<std::string> GetPaths() const { return path_to_pointclouds_; }
    std::string GetPath(size_t index) const {
        if (index > 2) {
            utility::LogError(
                    "Invalid index. Expected index between 0 to 2 but got {}.",
                    index);
        }
        return path_to_pointclouds_[index];
    }

private:
    // List of path to PCD point-cloud fragments.
    // path_to_pointclouds_[x] returns path to `cloud_bin_x.pcd` where x is from
    // 0 to 2.
    std::vector<std::string> path_to_pointclouds_;
};

class RedwoodLivingRoomFragments : public SimpleDataset {
public:
    RedwoodLivingRoomFragments(
            const std::string& prefix = "RedwoodLivingRoomFragments",
            const std::string& data_root = "")
        : SimpleDataset(prefix,
                        {"http://redwood-data.org/indoor/data/"
                         "livingroom1-fragments-ply.zip",
                         "https://github.com/isl-org/open3d_downloads/releases/"
                         "download/redwood/livingroom1-fragments-ply.zip"},
                        "36e0eb23a66ccad6af52c05f8390d33e") {
        path_to_pointclouds_.reserve(57);
        for (int i = 0; i < 57; ++i) {
            path_to_pointclouds_.push_back(Dataset::GetExtractDir() +
                                           "/cloud_bin_" + std::to_string(i) +
                                           ".ply");
        }
    }

    std::string GetPath(size_t index) const {
        if (index > 56) {
            utility::LogError(
                    "Invalid index. Expected index between 0 to 56 but got {}.",
                    index);
        }
        return path_to_pointclouds_[index];
    }
    std::vector<std::string> GetPaths() const { return path_to_pointclouds_; }

private:
    // Path to PLY point-cloud fragments.
    // path_to_pointclouds_[x] return path to `cloud_bin_x.ply` where x is from
    // 0 to 56.
    std::vector<std::string> path_to_pointclouds_;
};

class RedwoodOfficeFragments : public SimpleDataset {
public:
    RedwoodOfficeFragments(const std::string& prefix = "RedwoodOfficeFragments",
                           const std::string& data_root = "")
        : SimpleDataset(prefix,
                        {"http://redwood-data.org/indoor/data/"
                         "office1-fragments-ply.zip",
                         "https://github.com/isl-org/open3d_downloads/releases/"
                         "download/redwood/office1-fragments-ply.zip"},
                        "c519fe0495b3c731ebe38ae3a227ac25") {
        path_to_pointclouds_.reserve(57);
        for (int i = 0; i < 52; ++i) {
            path_to_pointclouds_.push_back(Dataset::GetExtractDir() +
                                           "/cloud_bin_" + std::to_string(i) +
                                           ".ply");
        }
    }

    std::vector<std::string> GetPaths() const { return path_to_pointclouds_; }
    std::string GetPath(size_t index) const {
        if (index > 51) {
            utility::LogError(
                    "Invalid index. Expected index between 0 to 51 but got {}.",
                    index);
        }
        return path_to_pointclouds_[index];
    }

private:
    // Path to PLY point-cloud fragments.
    // path_to_pointclouds_[x] return path to `cloud_bin_x.ply` where x is from
    // 0 to 51.
    std::vector<std::string> path_to_pointclouds_;
};

}  // namespace dataset
}  // namespace data
}  // namespace open3d
