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

#include "open3d/data/Download.h"
#include "open3d/data/Extract.h"
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

    // Initialize paths, to be used by other helper functions.
    download_prefix_ = "download/" + prefix_;
    path_to_download_ = data_root_ + "/" + download_prefix_;
    extract_prefix_ = "extract/" + prefix_;
    path_to_extract_ = data_root_ + "/" + extract_prefix_;
}

void Dataset::DeleteDownloadFiles() const {
    utility::filesystem::DeleteDirectory(path_to_download_);
}

void Dataset::DeleteExtractFiles() const {
    utility::filesystem::DeleteDirectory(path_to_extract_);
}

static bool VerifyFiles(const std::string& data_path,
                        const std::unordered_map<std::string, std::string>&
                                file_name_to_file_md5) {
    for (auto& file : file_name_to_file_md5) {
        const std::string file_path = data_path + "/" + file.first;
        if (!utility::filesystem::FileExists(file_path)) return false;
        if (GetMD5(file_path) != file.second) return false;
    }
    return true;
}

TemplateDataset::TemplateDataset(const std::string& prefix,
                                 const std::vector<std::string>& url_mirrors,
                                 const std::string& md5,
                                 const bool no_extract,
                                 const std::string& data_root)
    : Dataset(prefix, data_root) {
    const std::string filename =
            utility::filesystem::GetFileNameWithoutDirectory(url_mirrors[0]);

    const bool is_extract_present =
            utility::filesystem::DirectoryExists(path_to_extract_);

    if (!is_extract_present) {
        // // Check cached download.
        if (!VerifyFiles(path_to_download_, {{filename, md5}})) {
            DownloadFromMirrorURLs(url_mirrors, md5, download_prefix_,
                                   data_root_);
        }

        // Extract / Copy data.
        const std::string download_file_path =
                path_to_download_ + "/" + filename;
        if (!no_extract) {
            Extract(download_file_path, path_to_extract_);
        } else {
            utility::filesystem::MakeDirectoryHierarchy(path_to_extract_);
            utility::filesystem::CopyFile(download_file_path, path_to_extract_);
        }
    }
}

}  // namespace data
}  // namespace open3d
