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

#include "open3d/data/Extract.h"

#include "open3d/data/Dataset.h"
#include "open3d/data/Downloader.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Extract, ExtractFromZIP) {
    // Download the `test_data_00.zip` test data.
    std::string test_file_url =
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "data-manager/test_data_00.zip";
    std::string test_file_sha256sum =
            "66ea466a02532d61dbc457abf1408afeab360d7a35e15f1479ca91c25e838d30";
    data::DownloadFromURL(test_file_url, "", "", false, test_file_sha256sum);

    const std::string download_dir = data::LocateDataRoot();
    std::string file_path = download_dir + "/test_data_00.zip";

    // Extract the test zip file.
    EXPECT_NO_THROW(data::Extract(file_path, download_dir));
    std::string extracted_folder = download_dir + "/test_data";
    std::string output_file = extracted_folder + "/lena_color.jpg";

    // Check if the extracted file exists.
    EXPECT_TRUE(utility::filesystem::FileExists(output_file));

    // Delete test file.
    std::remove(output_file.c_str());
    utility::filesystem::DeleteDirectory(extracted_folder);
    std::remove(file_path.c_str());

    // Download the `test_data_00.tar.xy` test data.
    test_file_url =
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "data-manager/test_data_00.tar.xz";
    test_file_sha256sum =
            "e8072ac8c10b73a13a9b72642f3645985e74c3853a71d984d000020455c0b3b7";
    data::DownloadFromURL(test_file_url, "", "", false, test_file_sha256sum);

    // Currently only `.zip` files are supported.
    file_path = download_dir + "/test_data_00.tar.xz";
    EXPECT_ANY_THROW(data::Extract(file_path, download_dir));
    std::remove(file_path.c_str());
}

}  // namespace tests
}  // namespace open3d
