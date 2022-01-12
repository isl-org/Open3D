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

/// A dataset class locates the data root directory in the following order:
///
/// (a) User-specified by `data_root` when instantiating a dataset object.
/// (b) OPEN3D_DATA_ROOT environment variable.
/// (c) $HOME/open3d_data.
///
/// LocateDataRoot() shall be called when the user-specified data root is not
/// set, i.e. in case (b) and (c).
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
    Dataset(const std::string& prefix, const std::string& data_root = "");

    ~Dataset() {}

    void DeleteDownloadFiles() const;
    void DeleteExtractFiles() const;

    /// Get data root directory. The data root is set at construction time or
    /// automatically determined.
    const std::string GetDataRoot() const { return data_root_; }

protected:
    /// Open3D data root.
    std::string data_root_;

public:
    /// Dataset prefix.
    std::string prefix_;
    /// Dataset help string containing informations such as source,
    /// documentation link, functionalities, usage, licence, and other useful
    /// informations.
    std::string help_;

    // Derived members, for convinience.
protected:
    /// extract_prefix_ = "/extract/" + prefix_
    std::string extract_prefix_;
    /// download_prefix_ = "/download/" + prefix_
    std::string download_prefix_;
    /// download_prefix_ = data_root_ + "/download/" + prefix_
    std::string path_to_download_;
    /// path_to_extract_ = data_root_ + "/extract/" + prefix_
    std::string path_to_extract_;
};

class TemplateDataset : public Dataset {
public:
    TemplateDataset(const std::string& prefix,
                    const std::vector<std::string>& url_mirrors,
                    const std::string& md5,
                    const bool no_extract = false,
                    const std::string& data_root = "");

    ~TemplateDataset() {}
};

namespace dataset {

class SamplePCDFragments : public TemplateDataset {
public:
    SamplePCDFragments(const std::string& prefix = "SamplePCDFragments",
                       const std::string& data_root = "")
        : TemplateDataset(
                  prefix,
                  {"https://github.com/isl-org/open3d_downloads/releases/"
                   "download/sample-pcd-fragments/SamplePCDFragments.zip"},
                  "4d39442a86e9fe80c967a6c513d57442") {
        path_to_extract_ = Dataset::path_to_extract_;

        for (int i = 0; i < 3; ++i) {
            path_to_fragments_.push_back(path_to_extract_ + "/cloud_bin_" +
                                         std::to_string(i) + ".pcd");
        }

        // clang-format off
        Dataset::help_ = 
            "\n Colored point-cloud fragments of living-room from ICL-NUIM "
            "\n RGBD Benchmark Dataset in PCD format."
            "\n Information: "
            "\n  Type: Point cloud fragments [contains points, colors, normals, curvature]."
            "\n  Format: PCD Binary."
            "\n  Source: ICL-NUIM RGBD Benchmark Dataset."
            "\n  MD5: 4d39442a86e9fe80c967a6c513d57442"
            "\n  "
            "\n  Contents of SamplePCDFragments.zip:"
            "\n  SamplePCDFragments"
            "\n    ├── cloud_bin_0.pcd"
            "\n    ├── cloud_bin_1.pcd"
            "\n    ├── cloud_bin_2.pcd"
            "\n    └── init.log"
            "\n "
            "\n Application: Used in Open3D ICP registration demo examples."
            "\n "
            "\n Licence: The data is released under Creative Commons 3.0 (CC BY 3.0), "
            "\n          see http://creativecommons.org/licenses/by/3.0/.";
        // clang-format on
    }

    // Path to PCD point-cloud fragments.
    // path_to_fragments_[x] return path to `cloud_bin_x.pcd` where x is from 0
    // to 2.
    std::vector<std::string> path_to_fragments_;
};

class RedwoodLivingRoomFragments : public TemplateDataset {
public:
    RedwoodLivingRoomFragments(
            const std::string& prefix = "RedwoodLivingRoomFragments",
            const std::string& data_root = "")
        : TemplateDataset(
                  prefix,
                  {"http://redwood-data.org/indoor/data/"
                   "livingroom1-fragments-ply.zip",
                   "https://github.com/isl-org/open3d_downloads/releases/"
                   "download/redwood/livingroom1-fragments-ply.zip"},
                  "36e0eb23a66ccad6af52c05f8390d33e") {
        path_to_fragments_.reserve(57);
        for (int i = 0; i < 57; ++i) {
            path_to_fragments_.push_back(Dataset::path_to_extract_ +
                                         "/cloud_bin_" + std::to_string(i) +
                                         ".ply");
        }
    }

    // Path to PLY point-cloud fragments.
    // path_to_fragments_[x] return path to `cloud_bin_x.ply` where x is from 0
    // to 56.
    std::vector<std::string> path_to_fragments_;
};

class RedwoodOfficeFragments : public TemplateDataset {
public:
    RedwoodOfficeFragments(const std::string& prefix = "RedwoodOfficeFragments",
                           const std::string& data_root = "")
        : TemplateDataset(
                  prefix,
                  {"http://redwood-data.org/indoor/data/"
                   "office1-fragments-ply.zip",
                   "https://github.com/isl-org/open3d_downloads/releases/"
                   "download/redwood/office1-fragments-ply.zip"},
                  "c519fe0495b3c731ebe38ae3a227ac25") {
        path_to_fragments_.reserve(57);
        for (int i = 0; i < 52; ++i) {
            path_to_fragments_.push_back(Dataset::path_to_extract_ +
                                         "/cloud_bin_" + std::to_string(i) +
                                         ".ply");
        }
    }

    // Path to PLY point-cloud fragments.
    // path_to_fragments_[x] return path to `cloud_bin_x.ply` where x is from 0
    // to 51.
    std::vector<std::string> path_to_fragments_;
};

}  // namespace dataset
}  // namespace data
}  // namespace open3d
