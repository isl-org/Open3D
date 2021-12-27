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
#include "open3d/data/Download.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Downloader, DownloadAndVerify) {
    std::string url =
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "data-manager/test_data_00.zip";
    std::string sha256 =
            "66ea466a02532d61dbc457abf1408afeab360d7a35e15f1479ca91c25e838d30";

    std::string prefix = "temp_test";
    std::string file_dir = data::LocateDataRoot() + "/" + prefix;
    std::string file_path = file_dir + "/" + "test_data_00.zip";
    if (utility::filesystem::DirectoryExists(file_dir)) {
        EXPECT_TRUE(utility::filesystem::DeleteDirectory(file_dir));
    }

    // This download shall work.
    data::DownloadFromURL(url, sha256, prefix);
    EXPECT_TRUE(utility::filesystem::DirectoryExists(file_dir));
    EXPECT_TRUE(
            utility::filesystem::FileExists(file_dir + "/test_data_00.zip"));

    // This download shall be skipped (look at the message).
    data::DownloadFromURL(url, sha256, prefix);

    // Mismatch sha256.
    EXPECT_ANY_THROW(data::DownloadFromURL(
            url,
            "0000000000000000000000000000000000000000000000000000000000000000",
            prefix));

    EXPECT_TRUE(utility::filesystem::RemoveFile(file_path));
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(file_dir));
}

}  // namespace tests
}  // namespace open3d
