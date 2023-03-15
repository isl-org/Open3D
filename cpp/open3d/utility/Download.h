// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace open3d {
namespace utility {

/// \brief Computes MD5 Hash for the given file.
/// \param file_path Path to the file.
std::string GetMD5(const std::string& file_path);

/// \brief Download a file from URL. If a file already exists and the MD5 hash
/// matches, the download will be skipped.
///
/// \param url File URL. The saved file name will be the last part of the URL.
/// \param md5 MD5 checksum of the file. This is required as the same
/// URL may point to different files over time.
/// \param download_dir The file will be downloaded to download_dir.
/// \returns Path to downloaded file.
/// \throw std::runtime_error If the download fails.
std::string DownloadFromURL(const std::string& url,
                            const std::string& md5,
                            const std::string& download_dir);

/// \brief Download a file from list of mirror URLs. If a file already exists
/// and the MD5 hash matches, the download will be skipped. All file names must
/// be the same in mirrors.
///
/// \param mirrors List of file mirror URLs for the same file. The saved
/// file name will be the last part of the URL.
/// \param md5 MD5 checksum of the file. This is required as the same URL may
/// point to different files over time.
/// \param download_dir The file will be downloaded to download_dir.
/// \returns Path to downloaded file.
/// \throw std::runtime_error If the download fails.
std::string DownloadFromMirrors(const std::vector<std::string>& mirrors,
                                const std::string& md5,
                                const std::string& download_dir);

}  // namespace utility
}  // namespace open3d
