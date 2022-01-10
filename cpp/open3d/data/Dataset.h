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
    Dataset(const std::string& prefix = "", const std::string& data_root = "");

    ~Dataset() {}

    void DisplayDataTree(const int depth_level = 0) const;

    /// \brief Display dataset related information, such as source, type, data
    /// loader functionalities, usage, licence, and other useful informations.
    const std::string Help() const;

    void DeleteDownloadFiles() const;
    void DeleteExtractFiles() const;

    /// Get data root directory. The data root is set at construction time or
    /// automatically determined.
    const std::string GetDataRoot() const { return data_root_; }

public:
    std::vector<std::string> download_filenames_;

protected:
    /// Open3D data root.
    std::string data_root_;
    std::string prefix_;
    std::string extract_prefix_;
    std::string download_prefix_;
    std::string path_to_extract_;
    std::string path_to_download_;
};

class TemplateDataset : public Dataset {
public:
    TemplateDataset(
            const std::string& prefix,
            const std::unordered_map<std::string, std::vector<std::string>>&
                    md5_to_mirror_urls,
            const bool no_extract = false,
            const std::string& data_root = "");

    ~TemplateDataset() {}

private:
    const std::unordered_map<std::string, std::vector<std::string>>
            md5_to_mirror_urls_;
    std::unordered_map<std::string, std::string> filenames_to_md5_;
};

namespace dataset {

static std::unordered_map<std::string, std::vector<std::string>>
        mirrors_open3d_sample_data{
                {"919880e42a741a50ed8cda34129dc3d7",
                 {"https://github.com/isl-org/open3d_downloads/releases/"
                  "download/00.14.01_sample_data/"
                  "open3d_sample_data_00140100.zip"}}};

class Open3DSampleData : public TemplateDataset {
public:
    Open3DSampleData(const std::string& prefix = "Open3DSampleData",
                     const std::string& data_root = "")
        : TemplateDataset(prefix, mirrors_open3d_sample_data) {
        path_ = path_to_extract_;
    }

    std::string path_;
};

}  // namespace dataset

}  // namespace data
}  // namespace open3d
