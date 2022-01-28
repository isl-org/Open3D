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

Dataset::Dataset(const std::string& prefix,
                 const std::string& help_string,
                 const std::string& data_root)
    : prefix_(prefix), help_string_(help_string) {
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
                             const std::string& help_string,
                             const std::string& data_root)
    : Dataset(prefix, help_string, data_root) {
    const std::string filename =
            utility::filesystem::GetFileNameWithoutDirectory(urls[0]);

    const bool is_extract_present =
            utility::filesystem::DirectoryExists(Dataset::GetExtractDir());

    if (!is_extract_present) {
        // `download_dir` is relative path from `${data_root}`.
        const std::string download_dir = "download/" + GetPrefix();
        const std::string download_file_path =
                utility::DownloadFromURL(urls, md5, download_dir, data_root_);

        // Extract / Copy data.
        if (!no_extract) {
            utility::Extract(download_file_path, Dataset::GetExtractDir());
        } else {
            utility::filesystem::MakeDirectoryHierarchy(
                    Dataset::GetExtractDir());
            utility::filesystem::CopyFile(download_file_path,
                                          Dataset::GetExtractDir());
        }
    }
}

SampleICPPointClouds::SampleICPPointClouds(const std::string& prefix,
                                           const std::string& data_root)
    : SimpleDataset(
              prefix,
              {"https://github.com/isl-org/open3d_downloads/releases/"
               "download/sample-icp-pointclouds/SampleICPPointClouds.zip"},
              "3ee7a2631caa3c47a333972e3c4fb315") {
    for (int i = 0; i < 3; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".pcd");
    }

    this->help_string_ = std::string(R"""(
Colored point-cloud fragments of living-room from ICL-NUIM
RGBD Benchmark Dataset in PCD format.

Information:
- Type: Point cloud fragments [contains points, colors, normals, curvature].
- Format: PCD Binary.

Contents of SampleICPPointClouds.zip:
    SampleICPPointClouds
    ├── cloud_bin_0.pcd
    ├── cloud_bin_1.pcd
    └── cloud_bin_2.pcd

Source: ICL-NUIM RGBD Benchmark Dataset.
Licence: Creative Commons 3.0 (CC BY 3.0).

Download Mirror:
- https://github.com/isl-org/open3d_downloads/releases/download/sample-icp-pointclouds/SampleICPPointClouds.zip
MD5: 3ee7a2631caa3c47a333972e3c4fb315
     )""");
}

std::string SampleICPPointClouds::GetPaths(size_t index) const {
    if (index > 2) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 2 but got {}.",
                index);
    }
    return paths_[index];
}

RedwoodLivingRoomFragments::RedwoodLivingRoomFragments(
        const std::string& prefix, const std::string& data_root)
    : SimpleDataset(prefix,
                    {"http://redwood-data.org/indoor/data/"
                     "livingroom1-fragments-ply.zip",
                     "https://github.com/isl-org/open3d_downloads/releases/"
                     "download/redwood/livingroom1-fragments-ply.zip"},
                    "36e0eb23a66ccad6af52c05f8390d33e") {
    paths_.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".ply");
    }

    this->help_string_ = std::string(R"""(
Colored point-cloud fragments of living-room-1 from ICL-NUIM
RGBD Benchmark Dataset in PLY format.

Information:
- Type: Point cloud fragments [contains points, colors, normals, curvature].
- Format: PLY Binary.

Contents of livingroom1-fragments-ply.zip:
    RedwoodLivingRoomFragments
    ├── cloud_bin_0.ply
    ├── cloud_bin_1.ply
    │   ...
    └── cloud_bin_56.ply

Source: ICL-NUIM RGBD Benchmark Dataset.
Licence: Creative Commons 3.0 (CC BY 3.0).

Download Mirrors:
- http://redwood-data.org/indoor/data/livingroom1-fragments-ply.zip
- https://github.com/isl-org/open3d_downloads/releases/download/redwood/livingroom1-fragments-ply.zip
MD5: 36e0eb23a66ccad6af52c05f8390d33e
     )""");
}

std::string RedwoodLivingRoomFragments::GetPaths(size_t index) const {
    if (index > 56) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 56 but got {}.",
                index);
    }
    return paths_[index];
}

RedwoodOfficeFragments::RedwoodOfficeFragments(const std::string& prefix,
                                               const std::string& data_root)
    : SimpleDataset(prefix,
                    {"http://redwood-data.org/indoor/data/"
                     "office1-fragments-ply.zip",
                     "https://github.com/isl-org/open3d_downloads/releases/"
                     "download/redwood/office1-fragments-ply.zip"},
                    "c519fe0495b3c731ebe38ae3a227ac25") {
    paths_.reserve(53);
    for (int i = 0; i < 53; ++i) {
        paths_.push_back(Dataset::GetExtractDir() + "/cloud_bin_" +
                         std::to_string(i) + ".ply");
    }

    this->help_string_ = std::string(R"""(
Colored point-cloud fragments of office-1 from ICL-NUIM
RGBD Benchmark Dataset in PLY format.

Information:
- Type: Point cloud fragments [contains points, colors, normals, curvature].
- Format: PLY Binary.

Contents of office1-fragments-ply.zip:
    RedwoodOfficeFragments
    ├── cloud_bin_0.ply
    ├── cloud_bin_1.ply
    │   ...
    └── cloud_bin_52.ply

Source: ICL-NUIM RGBD Benchmark Dataset.
Licence: Creative Commons 3.0 (CC BY 3.0).

Download Mirrors:
- http://redwood-data.org/indoor/data/office1-fragments-ply.zip
- https://github.com/isl-org/open3d_downloads/releases/download/redwood/office1-fragments-ply.zip
MD5: 36e0eb23a66ccad6af52c05f8390d33e
     )""");
}

std::string RedwoodOfficeFragments::GetPaths(size_t index) const {
    if (index > 52) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 52 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
