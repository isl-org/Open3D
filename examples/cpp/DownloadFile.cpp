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

std::string JoinPath(const std::string prefix_path, const std::string path) {
    std::string output = prefix_path + "/" + path;
    return output;
}

void DownloadAndExtract(const std::string url,
                        const std::string expected_sha256) {
    const std::string filename_from_url = "test_file.zip";
    const std::string default_data_root =
            utility::filesystem::GetHomeDirectory() + "/open3d_data";

    const std::string filename_random = "random_name.zip";
    const std::string random_dir_hierarchy =
            utility::filesystem::GetHomeDirectory() +
            "/test_folder1/test_folder2";

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
    if (!data::DownloadFromURL(url)) {
        utility::LogInfo("Method 1 Failed");
    }
    std::string file_to_extract =
            JoinPath(default_data_root, filename_from_url);
    if (!data::Extract(file_to_extract, default_data_root)) {
        utility::LogInfo("Extraction Failed.");
    }

    // Download with custom name.
    // Download in Open3D Data Root directory,
    // with the the given name `random_name.zip`,
    // `always_download` is False : Skip download if file exists, with correct
    // SHA256SUM.
    // SHA256 is required. Not providing this, will throw Runtime ERROR.
    // Download in Open3D Data Root directory, with the given file name.
    if (!data::DownloadFromURL(url, "", filename_random, false,
                               expected_sha256)) {
        utility::LogInfo("Method 2 Failed");
    }
    file_to_extract = JoinPath(default_data_root, filename_random);
    if (!data::Extract(file_to_extract, default_data_root, "", true, true)) {
        utility::LogInfo("Extraction Failed.");
    }

    // Download in specified directory (creates the directory hierarchy if not
    // present), with the original file name extracted from the url.
    if (!data::DownloadFromURL(url, random_dir_hierarchy, "", false,
                               expected_sha256)) {
        utility::LogInfo("Method 3 Failed");
    }
    file_to_extract = JoinPath(random_dir_hierarchy, filename_from_url);
    if (!data::Extract(file_to_extract, random_dir_hierarchy, "", true, true)) {
        utility::LogInfo("Extraction Failed.");
    }

    // Download in specified directory (creates the directory hierarchy if not
    // present), with the given name.
    if (!data::DownloadFromURL(url, random_dir_hierarchy, filename_random,
                               false, expected_sha256)) {
        utility::LogInfo("Method 4 Failed");
    }
    file_to_extract = JoinPath(random_dir_hierarchy, filename_random);
    if (!data::Extract(file_to_extract, random_dir_hierarchy, "", true, true)) {
        utility::LogInfo("Extraction Failed.");
    }

    // Print calculated SHA256SUM.
    auto file_actual_SHA256 =
            data::GetSHA256(random_dir_hierarchy + "/random_name.zip");

    utility::LogInfo("SHA256SUM: {}", file_actual_SHA256);
}

int main(int argc, char *argv[]) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    const std::string url =
            "https://github.com/reyanshsolis/rey_download/releases/download/"
            "test_data/test_file.zip";

    const std::string expected_sha256 =
            "844c677b4bbf9e63035331769947ada46640187ac4caeff50f22c14f76e5f814";

    DownloadAndExtract(url, expected_sha256);

    return 0;
}
