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
}

SimpleDataset::SimpleDataset(const std::string& prefix,
                             const std::vector<std::string>& urls,
                             const std::string& md5,
                             const bool no_extract,
                             const std::string& data_root)
    : Dataset(prefix, data_root) {
    const std::string filename =
            utility::filesystem::GetFileNameWithoutDirectory(urls[0]);

    const bool is_extract_present =
            utility::filesystem::DirectoryExists(Dataset::GetExtractDir());

    if (!is_extract_present) {
        const std::string download_file_path = DownloadFromURL(
                urls, md5,
                Dataset::GetDownloadDir(/*relative_to_data_root*/ true),
                data_root_);

        // Extract / Copy data.
        if (!no_extract) {
            Extract(download_file_path, Dataset::GetExtractDir());
        } else {
            utility::filesystem::MakeDirectoryHierarchy(
                    Dataset::GetExtractDir());
            utility::filesystem::CopyFile(download_file_path,
                                          Dataset::GetExtractDir());
        }
    }
}

}  // namespace data
}  // namespace open3d
