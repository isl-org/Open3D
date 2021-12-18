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

#include "open3d/data/Downloader.h"

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Downloader, DownloadAndVerify) {
    // File 1.
    std::string url =
            "https://github.com/isl-org/open3d_downloads/releases/download/"
            "data-manager/test_data_00.zip";
    std::string expected_sha256 =
            "66ea466a02532d61dbc457abf1408afeab360d7a35e15f1479ca91c25e838d30";

    // Downloader API.
    // URL is the only compulsory input, others are optional, and passing
    // empty string "", trigers the default behaviour.
    // open3d::data::DownloadFromURL(url, output_file_name, output_file_path,
    //                               always_download, expected_sha256);

    // Default.
    // Download in Open3D Data Root directory,
    // with the original file name extracted from the url,
    // `always_download` is True : If exists, it will be over-written.
    // SHA256 is not verified.
    EXPECT_TRUE(data::DownloadFromURL(url));
    EXPECT_TRUE(data::DownloadFromURL(url, "", "", false, expected_sha256));
}

}  // namespace tests
}  // namespace open3d
