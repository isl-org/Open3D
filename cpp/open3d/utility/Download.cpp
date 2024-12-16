// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Download.h"

// BoringSSL is currently compatible with OpenSSL 1.1.0
#define OPENSSL_API_COMPAT 10100
// clang-format off
// Must include openssl before curl to build on Windows.
#include <openssl/md5.h>

// https://stackoverflow.com/a/41873190/1255535
#ifdef WINDOWS
#pragma comment(lib, "wldap32.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "Ws2_32.lib")
#define USE_SSLEAY
#define USE_OPENSSL
#endif

#define CURL_STATICLIB

#include <curl/curl.h>
#include <curl/easy.h>
// clang-format on

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

#include "open3d/data/Dataset.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

std::string GetMD5(const std::string& file_path) {
    if (!utility::filesystem::FileExists(file_path)) {
        utility::LogError("{} does not exist.", file_path);
    }

    std::ifstream fp(file_path.c_str(), std::ios::in | std::ios::binary);

    if (!fp.good()) {
        std::ostringstream os;
        utility::LogError("Cannot open {}", file_path);
    }

    constexpr const std::size_t buffer_size{1 << 12};  // 4 KiB
    char buffer[buffer_size];
    unsigned char hash[MD5_DIGEST_LENGTH] = {0};

    MD5_CTX ctx;
    MD5_Init(&ctx);

    while (fp.good()) {
        fp.read(buffer, buffer_size);
        MD5_Update(&ctx, buffer, fp.gcount());
    }

    MD5_Final(hash, &ctx);
    fp.close();

    std::ostringstream os;
    os << std::hex << std::setfill('0');

    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        os << std::setw(2) << static_cast<unsigned int>(hash[i]);
    }

    return os.str();
}

static size_t WriteDataCb(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
}

std::string DownloadFromURL(const std::string& url,
                            const std::string& md5,
                            const std::string& download_dir) {
    // Sanity checks.
    if (md5.size() != MD5_DIGEST_LENGTH * 2) {
        utility::LogError("Invalid md5 length {}, expected to be {}.",
                          md5.size(), MD5_DIGEST_LENGTH * 2);
    }
    if (download_dir.empty()) {
        utility::LogError("download_dir cannot be empty.");
    }

    // Resolve path.
    const std::string file_name =
            utility::filesystem::GetFileNameWithoutDirectory(url);
    const std::string file_path = download_dir + "/" + file_name;
    if (!utility::filesystem::DirectoryExists(download_dir)) {
        utility::filesystem::MakeDirectoryHierarchy(download_dir);
    }

    // Check if the file exists.
    if (utility::filesystem::FileExists(file_path) &&
        GetMD5(file_path) == md5) {
        utility::LogInfo("{} exists and MD5 matches. Download skipped.",
                         file_path);
        return file_path;
    }

    // Always print URL to inform the user. If the download fails, the user
    // knows the URL.
    utility::LogInfo("Downloading {}", url);

    // Download.
    CURL* curl;
    FILE* fp;
    CURLcode res;
    curl = curl_easy_init();
    if (!curl) {
        utility::LogError("Failed to initialize CURL.");
    }
    fp = fopen(file_path.c_str(), "wb");
    if (!fp) {
        utility::LogError("Failed to open file {}.", file_path);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // -L redirection.
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, false);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteDataCb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);

    if (res == CURLE_OK) {
        const std::string actual_md5 = GetMD5(file_path);
        if (actual_md5 == md5) {
            utility::LogInfo("Downloaded to {}", file_path);
        } else {
            utility::LogError(
                    "MD5 mismatch for {}.\n- Expected: {}\n- Actual  : {}",
                    file_path, md5, actual_md5);
        }
    } else {
        utility::LogError("Download failed with error code: {}.",
                          curl_easy_strerror(res));
    }

    return file_path;
}

std::string DownloadFromMirrors(const std::vector<std::string>& mirrors,
                                const std::string& md5,
                                const std::string& download_dir) {
    // All file names must be the same in mirrors.
    if (mirrors.empty()) {
        utility::LogError("No mirror URLs provided.");
    }
    const std::string file_name =
            utility::filesystem::GetFileNameWithoutDirectory(mirrors[0]);
    for (const std::string& url : mirrors) {
        if (utility::filesystem::GetFileNameWithoutDirectory(url) !=
            file_name) {
            utility::LogError("File name mismatch in mirrors {}.", mirrors);
        }
    }

    for (const std::string& url : mirrors) {
        try {
            return DownloadFromURL(url, md5, download_dir);
        } catch (const std::exception& ex) {
            utility::LogWarning("Failed to download from {}. Exception {}.",
                                url, ex.what());
        }
    }

    utility::LogError("Downloading failed from available mirrors.");
}

}  // namespace utility
}  // namespace open3d
