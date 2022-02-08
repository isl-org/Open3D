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

#include "open3d/utility/Download.h"

#include "open3d/data/Dataset.h"
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
    std::string md5 = "996987b27c4497dbb951ec056c9684f4";

    std::string prefix = "temp_test";
    std::string file_dir = data::LocateDataRoot() + "/" + prefix;
    std::string file_path = file_dir + "/" + "test_data_00.zip";
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(file_dir));

    // This download shall work.
    EXPECT_EQ(utility::DownloadFromURL(url, md5, prefix), file_path);
    EXPECT_TRUE(utility::filesystem::DirectoryExists(file_dir));
    EXPECT_TRUE(utility::filesystem::FileExists(file_path));
    EXPECT_EQ(utility::GetMD5(file_path), md5);

    // This download shall be skipped as the file already exists (look at the
    // message).
    EXPECT_EQ(utility::DownloadFromURL(url, md5, prefix), file_path);

    // Mismatch md5.
    EXPECT_ANY_THROW(utility::DownloadFromURL(
            url, "00000000000000000000000000000000", prefix));

    // Clean up.
    EXPECT_TRUE(utility::filesystem::DeleteDirectory(file_dir));
}

}  // namespace tests
}  // namespace open3d
