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

#include "open3d/Open3D.h"

using namespace open3d;

int main() {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    const std::string url =
            "https://github.com/reyanshsolis/rey_download/releases/download/"
            "test_data/test_file.zip";

    const std::string Expected_SHA256 =
            "844c677b4bbf9e63035331769947ada46640187ac4caeff50f22c14f76e5f814";

    const std::string random_dir_hierarchy =
            utility::filesystem::GetHomeDirectory() +
            "/test_folder1/test_folder2";

    data::Downloader downloader;

    // Downloader API.
    // URL is the only compulsory input, others are optional, and passing
    // empty string "", trigers the default behaviour.
    // downloader.DownloadFromURL(url, output_file_name, output_file_path,
    //                            expected_sha256);

    // Download in Open3D Data Root directory, with the original file name
    // extracted from the url.
    if (!downloader.DownloadFromURL(url)) {
        utility::LogInfo("Method 1 Failed");
    }

    // Download in Open3D Data Root directory, with the given file name.
    if (!downloader.DownloadFromURL(url, "", "random_name.zip",
                                    Expected_SHA256)) {
        utility::LogInfo("Method 2 Failed");
    }

    // Download in specified directory (creates the directory hierarchy if not
    // present), with the original file name extracted from the url.
    if (!downloader.DownloadFromURL(url, random_dir_hierarchy, "",
                                    Expected_SHA256)) {
        utility::LogInfo("Method 3 Failed");
    }

    // Download in specified directory (creates the directory hierarchy if not
    // present), with the given name.
    if (downloader.DownloadFromURL(url, random_dir_hierarchy, "random_name.zip",
                                   Expected_SHA256)) {
        auto file_Expected_SHA256 =
                data::GetSHA256(random_dir_hierarchy + "/random_name.zip");

        utility::LogInfo("Expected_SHA256: {}", file_Expected_SHA256);
    } else {
        utility::LogInfo("Method 4 Failed");
    }

    return 0;
}
