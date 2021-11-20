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

#include <curl/curl.h>
#include <curl/easy.h>
// #include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>

// #include <fstream>
// #include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

// #define CURL_STATICLIB

namespace open3d {
namespace data {

// REPLICATED FUNCTION FROM DATASET.h. TO BE REFACTORED.
/// A dataset class locates the data root directory in the following order:
///
/// (a) User-specified by `data_root` when instantiating a dataset object.
/// (b) OPEN3D_DATA_ROOT environment variable.
/// (c) $HOME/open3d_data.
///
/// LocateDataRoot() shall be called when the user-specified data root is not
/// set, i.e. in case (b) and (c).
static std::string LocateDataRoot() {
    std::string data_root = "";
    if (const char* env_p = std::getenv("OPEN3D_DATA_ROOT")) {
        data_root = std::string(env_p);
    }
    if (data_root.empty()) {
        data_root = utility::filesystem::GetHomeDirectory() + "/open3d_data";
    }
    return data_root;
}

Downloader::Downloader(const std::string& data_root) {
    if (data_root.empty()) {
        data_root_ = LocateDataRoot();
    } else {
        data_root_ = data_root;
    }
    utility::LogDebug("Downloader: Open3D Data Root is at {}", data_root_);
}

std::string Downloader::GetDataRoot() const { return data_root_; }

static size_t WriteDataCb(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
}

static std::string ExtractFileNameFromURL(const std::string& url) {
    int i = url.size();
    for (; i >= 0; i--) {
        if (url[i] == '/') {
            break;
        }
    }

    return url.substr(i + 1, url.size() - 1);
}

static std::string GetAbsoluteFilePath(const std::string& url,
                                       const std::string& output_file_path,
                                       const std::string& output_file_name) {
    std::string file_name;
    if (output_file_name.empty()) {
        file_name = ExtractFileNameFromURL(url);
    } else {
        file_name = output_file_name;
    }

    std::string file_path;
    if (output_file_path.empty()) {
        file_path = LocateDataRoot();
    } else {
        file_path = output_file_path;
    }

    // It will create the directory hierarchy if not present.
    if (!utility::filesystem::DirectoryExists(file_path)) {
        utility::filesystem::MakeDirectoryHierarchy(file_path);
    }

    file_path = file_path + '/' + file_name;
    return file_path;
}

bool Downloader::DownloadFromURL(const std::string& url,
                                 const std::string& output_file_path,
                                 const std::string& output_file_name) {
    std::string file_path =
            GetAbsoluteFilePath(url, output_file_path, output_file_name);

    CURL* curl;
    FILE* fp;
    CURLcode res;

    curl = curl_easy_init();
    if (curl) {
        fp = fopen(file_path.c_str(), "wb");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Follow redirection in link.
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, false);

        // Write function callback.
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteDataCb);

        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        // curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION,
        //                  download_progress_callback);
        // curl_easy_setopt(curl, CURLOPT_XFERINFODATA,
        //                  static_cast<void*>(&progress_bar));
        // curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);

        // Perform a file transfer synchronously.
        res = curl_easy_perform(curl);
        // Cleanup.
        curl_easy_cleanup(curl);
        // Close file.
        fclose(fp);

        if (res == CURLE_OK) {
            return true;
        } else {
            return false;
        }

    } else {
        return false;
    }
}

std::string Downloader::GetSHA256(const std::string& filename) {
    // const int K_READ_BUF_SIZE{1024 * 16};

    std::string empty_string = "";

    // // Initialize openssl
    // SHA256_CTX context;
    // if (!SHA256_Init(&context)) {
        return empty_string;
    // }

    // // Read file and update calculated SHA
    // char buf[K_READ_BUF_SIZE];
    // std::ifstream file(filename, std::ifstream::binary);
    // while (file.good()) {
    //     file.read(buf, sizeof(buf));
    //     if (!SHA256_Update(&context, buf, file.gcount())) {
    //         return empty_string;
    //     }
    // }

    // // Get Final SHA
    // unsigned char result[SHA256_DIGEST_LENGTH];
    // if (!SHA256_Final(result, &context)) {
    //     return empty_string;
    // }

    // // Transform byte-array to string
    // std::stringstream shastr;
    // shastr << std::hex << std::setfill('0');
    // for (const auto& byte : result) {
    //     shastr << std::setw(2) << (int)byte;
    // }
    // return shastr.str();
}

}  // namespace data
}  // namespace open3d
