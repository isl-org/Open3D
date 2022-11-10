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

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Dataset, DatasetBase) {
    data::Dataset ds("some_prefix");
    EXPECT_EQ(ds.GetDataRoot(),
              utility::filesystem::GetHomeDirectory() + "/open3d_data");

    data::Dataset ds_custom("some_prefix", "/my/custom/data_root");

    EXPECT_EQ(ds_custom.GetPrefix(), "some_prefix");
    EXPECT_EQ(ds_custom.GetDataRoot(), "/my/custom/data_root");
    EXPECT_EQ(ds_custom.GetDownloadDir(),
              "/my/custom/data_root/download/some_prefix");
    EXPECT_EQ(ds_custom.GetExtractDir(),
              "/my/custom/data_root/extract/some_prefix");
}

TEST(Dataset, DownloadDataset) {
    const std::string prefix = "DownloadDataset";
    const std::string data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";
    const std::string download_dir = data_root + "/download/" + prefix;
    const std::string extract_dir = data_root + "/extract/" + prefix;

    const std::vector<std::string> url_mirrors = {
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "20220201-data/BunnyMesh.ply"};
    const std::string md5 = "568f871d1a221ba6627569f1e6f9a3f2";

    data::DownloadDataset single_download_dataset(prefix, {url_mirrors, md5},
                                                  data_root);

    EXPECT_TRUE(
            utility::filesystem::FileExists(download_dir + "/BunnyMesh.ply"));
    EXPECT_TRUE(
            utility::filesystem::FileExists(extract_dir + "/BunnyMesh.ply"));

    EXPECT_EQ(single_download_dataset.GetPrefix(), prefix);
    EXPECT_EQ(single_download_dataset.GetDataRoot(), data_root);
    EXPECT_EQ(single_download_dataset.GetDownloadDir(), download_dir);
    EXPECT_EQ(single_download_dataset.GetExtractDir(), extract_dir);
}

}  // namespace tests
}  // namespace open3d
