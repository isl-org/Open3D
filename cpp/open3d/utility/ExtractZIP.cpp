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

#include "open3d/utility/ExtractZIP.h"

// Reference:
// https://github.com/madler/zlib/blob/master/contrib/minizip/miniunz.c

#include <errno.h>
#include <stdio.h>
#include <unzip.h>

#include <iostream>
#include <string>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

#ifdef __APPLE__
// In darwin and perhaps other BSD variants off_t is a 64 bit value, hence no
// need for specific 64 bit functions
#define FOPEN_FUNC(filename, mode) fopen(filename, mode)
#else
#define FOPEN_FUNC(filename, mode) fopen64(filename, mode)
#endif

// If required in future, the `WRITEBUFFERSIZE` size can be increased to 16384.
#define WRITEBUFFERSIZE (8192)

namespace open3d {
namespace utility {

static int ExtractCurrentFile(unzFile uf,
                              const std::string &extract_dir,
                              const std::string &password) {
    char filename_inzip[256];
    char *filename_withoutpath;
    char *p;
    int err = UNZ_OK;
    FILE *fout = nullptr;
    void *buf;
    uInt size_buf;

    unz_file_info64 file_info;
    err = unzGetCurrentFileInfo64(uf, &file_info, filename_inzip,
                                  sizeof(filename_inzip), nullptr, 0, nullptr,
                                  0);

    if (err != UNZ_OK) {
        utility::LogWarning("Error {} with zipfile in unzGetCurrentFileInfo.",
                            err);
        return err;
    }

    size_buf = WRITEBUFFERSIZE;
    buf = (void *)malloc(size_buf);
    if (buf == nullptr) {
        utility::LogWarning("Error allocating memory.");
        return UNZ_INTERNALERROR;
    }

    //  If zip entry is a directory then create it on disk.
    p = filename_withoutpath = filename_inzip;
    while ((*p) != '\0') {
        if (((*p) == '/') || ((*p) == '\\')) {
            filename_withoutpath = p + 1;
        }
        p++;
    }

    if ((*filename_withoutpath) == '\0') {
        const std::string dir_path = extract_dir + "/" + filename_inzip;
        utility::LogDebug("Creating directory: {}", dir_path);
        utility::filesystem::MakeDirectoryHierarchy(dir_path);
    } else {
        const char *write_filename;
        write_filename = filename_inzip;

        if (password.empty()) {
            err = unzOpenCurrentFilePassword(uf, nullptr);
        } else {
            err = unzOpenCurrentFilePassword(uf, password.c_str());
        }

        if (err != UNZ_OK) {
            utility::LogWarning(
                    "Extraction failed in unzOpenCurrentFilePassword with "
                    "error code: {}.",
                    err);
            return err;
        }

        if (err == UNZ_OK) {
            std::string file_path = extract_dir + "/" +
                                    static_cast<std::string>(write_filename);
            fout = FOPEN_FUNC(file_path.c_str(), "wb");

            // Some zipfile don't contain directory alone before file.
            if ((fout == nullptr) &&
                filename_withoutpath == (char *)filename_inzip) {
                utility::filesystem::MakeDirectoryHierarchy(extract_dir);

                fout = FOPEN_FUNC(file_path.c_str(), "wb");
            }

            if (fout == nullptr) {
                utility::LogWarning("Extraction failed. Error opening {}",
                                    file_path);
                return UNZ_ERRNO;
            }
        }

        if (fout != nullptr) {
            utility::LogDebug(" Extracting: {}", write_filename);

            do {
                err = unzReadCurrentFile(uf, buf, size_buf);
                if (err < 0) {
                    utility::LogWarning(
                            "Extraction failed in unzReadCurrentFile with "
                            "error code: {}.",
                            err);
                    break;
                }
                if (err > 0)
                    if (fwrite(buf, err, 1, fout) != 1) {
                        utility::LogWarning(
                                "Extraction failed. Error in writing extracted "
                                "file.");
                        err = UNZ_ERRNO;
                        break;
                    }
            } while (err > 0);

            if (fout) {
                fclose(fout);
            }
        }

        if (err == UNZ_OK) {
            err = unzCloseCurrentFile(uf);
            if (err != UNZ_OK) {
                utility::LogWarning(
                        "Extraction failed in unzCloseCurrentFile with error "
                        "code: {}.",
                        err);
            }
        } else {
            unzCloseCurrentFile(uf);
        }
    }

    free(buf);
    return err;
}

void ExtractFromZIP(const std::string &file_path,
                    const std::string &extract_dir) {
    unzFile uf = nullptr;
    if (!file_path.empty()) {
        uf = unzOpen64(file_path.c_str());
    }
    if (uf == nullptr) {
        utility::LogError("Failed to open file {}.", file_path);
    }

    unz_global_info64 gi;
    int err = unzGetGlobalInfo64(uf, &gi);
    if (err != UNZ_OK) {
        // Close file, before throwing exception.
        unzClose(uf);
        utility::LogError(
                "Extraction failed in unzGetGlobalInfo with error code: {}.",
                err);
    }

    // ExtractFromZIP supports password. Can be exposed if required in future.
    const std::string password = "";

    for (uLong i = 0; i < gi.number_entry; ++i) {
        err = ExtractCurrentFile(uf, extract_dir, password);
        if (err != UNZ_OK) {
            // Close file, before throwing exception.
            unzClose(uf);
            utility::LogError(
                    "Extraction failed in ExtractCurrentFile with error code: "
                    "{}.",
                    err);
        }

        if ((i + 1) < gi.number_entry) {
            err = unzGoToNextFile(uf);
            if (err != UNZ_OK) {
                // Close file, before throwing exception.
                unzClose(uf);
                utility::LogError(
                        "Extraction failed in ExtractCurrentFile with error "
                        "code: {}.",
                        err);
            }
        }
    }

    // Extracted Successfully. Close File.
    unzClose(uf);
}

}  // namespace utility
}  // namespace open3d
