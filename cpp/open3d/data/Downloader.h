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

#pragma once

#include <string>

namespace open3d {
namespace data {

/// \brief Computes SHA256 Hash for the given file.
/// \param file_path Path to the file.
std::string GetSHA256(const std::string& file_path);

/// \class Downloader
/// \brief Open3D data downloader class.
///
/// The downloader helps Open3D dataset class to download files and verify
/// them. The downloaded data will be stored in the Open3D's data
/// root directory.
///
/// - Downloader class locates the data root directory in the following order:
///   (a) User-specified by `data_root` when instantiating a downloader object.
///   (b) OPEN3D_DATA_ROOT environment variable.
///   (c) $HOME/open3d_data.
///   By default, (c) will be used, and it is also the recommended way.
class Downloader {
public:
    Downloader(const std::string& data_root = "");
    ~Downloader() {}

    /// Get data root directory. The data root is set at construction time or
    /// automatically determined.
    std::string GetDataRoot() const;

    /// \brief Function to download the file from URL.
    /// \param url URL for the file to be downloaded.
    /// \param output_file_path Custom directory to download the file. If
    /// directory does not exists, it will be created. If empty string is
    /// passed, the default data-root will be used.
    /// \param output_file_name Name of the downloaded file. If empty string is
    /// passed, the default file name will be used, extracted from the url.
    /// \param SHA256 SHA256SUM HASH value to verify the file after download. If
    /// empty string is passed, the verification will be skipped.
    bool DownloadFromURL(const std::string& url,
                         const std::string& output_file_path = "",
                         const std::string& output_file_name = "",
                         const std::string& SHA256 = "");

protected:
    /// Open3D data root.
    std::string data_root_;
};

}  // namespace data
}  // namespace open3d
