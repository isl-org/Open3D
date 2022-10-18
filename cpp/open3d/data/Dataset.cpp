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

#include "open3d/data/Dataset.h"

#include <string>

#include "open3d/utility/Download.h"
#include "open3d/utility/Extract.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

std::string LocateDataRoot() {
    std::string data_root = "";
    if (const char* env_p = std::getenv("OPEN3D_DATA_ROOT")) {
        data_root = std::string(env_p);
    }
    if (data_root.empty()) {
        data_root = utility::filesystem::GetHomeDirectory() + "/open3d_data";
    }
    return data_root;
}

std::string Open3DDownloadsPrefix() {
    return "https://github.com/isl-org/open3d_downloads/releases/download/";
}

Dataset::Dataset(const std::string& prefix, const std::string& data_root)
    : prefix_(prefix) {
    if (data_root.empty()) {
        data_root_ = LocateDataRoot();
    } else {
        data_root_ = data_root;
    }
    if (prefix_.empty()) {
        utility::LogError("prefix cannot be empty.");
    }
}

void Dataset::CheckPathsExist(const std::vector<std::string>& paths) const {
    const size_t num_expected_paths = paths.size();
    size_t num_existing_paths = 0;
    for (const auto& path : paths) {
        if (utility::filesystem::FileExists(path)) {
            num_existing_paths++;
        } else {
            utility::LogWarning("{} does not exist.", path);
        }
    }
    if (num_existing_paths != num_expected_paths) {
        utility::LogError(
                "Expected {} files, but only found {} files. Please "
                "re-download and re-extract the dataset.",
                num_expected_paths, num_existing_paths);
    }
}

DownloadDataset::DownloadDataset(const std::string& prefix,
                                 const DataDescriptor& data_descriptor,
                                 const std::string& data_root)
    : DownloadDataset(
              prefix, std::vector<DataDescriptor>{data_descriptor}, data_root) {
}

DownloadDataset::DownloadDataset(
        const std::string& prefix,
        const std::vector<DataDescriptor>& data_descriptors,
        const std::string& data_root)
    : Dataset(prefix, data_root), data_descriptors_(data_descriptors) {
    // Download.
    for (const auto& data_descriptor : data_descriptors) {
        if (!HasDownloaded(data_descriptor)) {
            utility::DownloadFromMirrors(data_descriptor.urls_,
                                         data_descriptor.md5_,
                                         GetDownloadDir());
            if (!HasDownloaded(data_descriptor)) {
                utility::LogError("Download failed integrity check.");
            }
        }
    }

    // Extract.
    // TODO: All dataset constructors should have full knowledge of the list of
    // extracted files. If CheckPathsExist() fails, we shall trigger
    // re-extraction automatically or throw an error. To enable this, we need to
    // modify the dataset constructors to provide a list of extracted files for
    // each dataset.
    const std::string base_extract_dir = GetExtractDir();
    if (!utility::filesystem::DirectoryExists(base_extract_dir)) {
        for (const auto& data_descriptor : data_descriptors) {
            const std::string download_name =
                    utility::filesystem::GetFileNameWithoutDirectory(
                            data_descriptor.urls_[0]);
            const std::string download_path =
                    GetDownloadDir() + "/" + download_name;

            std::string extract_dir = base_extract_dir;
            if (!data_descriptor.extract_in_subdir_.empty()) {
                extract_dir += "/" + data_descriptor.extract_in_subdir_;
            }

            if (utility::IsSupportedCompressedFilePath(download_path)) {
                utility::Extract(download_path, extract_dir);
            } else {
                utility::filesystem::MakeDirectoryHierarchy(extract_dir);
                utility::filesystem::Copy(download_path, extract_dir);
            }
        }
    }
}

bool DownloadDataset::HasDownloaded(
        const DataDescriptor& data_descriptor) const {
    // Check directory.
    if (!utility::filesystem::DirectoryExists(GetDownloadDir())) {
        return false;
    }
    // Check file exists.
    const std::string download_path =
            GetDownloadDir() + "/" +
            utility::filesystem::GetFileNameWithoutDirectory(
                    data_descriptor.urls_[0]);
    if (!utility::filesystem::FileExists(download_path)) {
        return false;
    }
    // Check MD5.
    if (utility::GetMD5(download_path) != data_descriptor.md5_) {
        return false;
    }
    return true;
}

}  // namespace data
}  // namespace open3d
