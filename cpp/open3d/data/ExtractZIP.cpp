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

#include "open3d/data/ExtractZIP.h"

// Reference:
// https://github.com/madler/zlib/blob/master/contrib/minizip/miniunz.c

#ifdef __APPLE__
// In darwin and perhaps other BSD variants off_t is a 64 bit value, hence no
// need for specific 64 bit functions
#define FOPEN_FUNC(filename, mode) fopen(filename, mode)
#else
#define FOPEN_FUNC(filename, mode) fopen64(filename, mode)
#endif

#include <unzip.h>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

#define CASESENSITIVITY (0)
// If required in future, the `WRITEBUFFERSIZE` size can be increased to 16384.
#define WRITEBUFFERSIZE (8192)
#define MAXFILENAME (256)

namespace open3d {
namespace data {

static int ExtractCurrentFile(unzFile uf, const std::string &password) {
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
        utility::LogError("Error {} with zipfile in unzGetCurrentFileInfo.",
                          err);
    }

    size_buf = WRITEBUFFERSIZE;
    buf = (void *)malloc(size_buf);
    if (buf == nullptr) {
        utility::LogError("Error allocating memory.");
    }

    //  If zip entry is a directory then create it on disk.
    p = filename_withoutpath = filename_inzip;
    while ((*p) != '\0') {
        if (((*p) == '/') || ((*p) == '\\')) filename_withoutpath = p + 1;
        p++;
    }

    if ((*filename_withoutpath) == '\0') {
        utility::LogDebug("Creating directory: {}", filename_inzip);
        utility::filesystem::MakeDirectoryHierarchy(
                std::string(filename_inzip));
    } else {
        const char *write_filename;
        write_filename = filename_inzip;

        if (password.empty()) {
            err = unzOpenCurrentFilePassword(uf, nullptr);
        } else {
            err = unzOpenCurrentFilePassword(uf, password.c_str());
        }

        if (err != UNZ_OK) {
            utility::LogError(
                    "Error {} with zipfile in unzOpenCurrentFilePassword.",
                    err);
        }

        if (err == UNZ_OK) {
            fout = FOPEN_FUNC(write_filename, "wb");

            // Some zipfile don't contain directory alone before file.
            if ((fout == nullptr) &&
                (filename_withoutpath != (char *)filename_inzip)) {
                char c = *(filename_withoutpath - 1);
                *(filename_withoutpath - 1) = '\0';

                utility::filesystem::MakeDirectoryHierarchy(
                        std::string(filename_inzip));

                *(filename_withoutpath - 1) = c;
                fout = FOPEN_FUNC(write_filename, "wb");
            }

            if (fout == nullptr) {
                utility::LogError("Error opening {}", write_filename);
            }
        }

        if (fout != nullptr) {
            utility::LogDebug(" Extracting: {}", write_filename);

            do {
                err = unzReadCurrentFile(uf, buf, size_buf);
                if (err < 0) {
                    utility::LogError(
                            "Error {} with zipfile in unzReadCurrentFile.",
                            err);
                }
                if (err > 0)
                    if (fwrite(buf, err, 1, fout) != 1) {
                        utility::LogError("Error in writing extracted file.");
                    }
            } while (err > 0);

            if (fout) {
                fclose(fout);
            }
        }

        if (err == UNZ_OK) {
            err = unzCloseCurrentFile(uf);
            if (err != UNZ_OK) {
                utility::LogError(
                        "Error {} with zipfile in unzCloseCurrentFile.", err);
            }
        } else {
            unzCloseCurrentFile(uf);
        }
    }

    free(buf);
    return err;
}

static void ExtractAll(unzFile uf, const std::string &password) {
    uLong i;
    unz_global_info64 gi;
    int err;

    err = unzGetGlobalInfo64(uf, &gi);
    if (err != UNZ_OK) {
        utility::LogError("Error {} with zipfile in unzGetGlobalInfo.", err);
    }

    for (i = 0; i < gi.number_entry; i++) {
        if (ExtractCurrentFile(uf, password) != UNZ_OK) {
            utility::LogError("Extraction failed in ExtractCurrentFile");
        }

        if ((i + 1) < gi.number_entry) {
            err = unzGoToNextFile(uf);
            if (err != UNZ_OK) {
                utility::LogError("Error {} with zipfile in unzGoToNextFile.",
                                  err);
            }
        }
    }
}

void ExtractFromZIP(const std::string &filename,
                    const std::string &extract_dir) {
    unzFile uf = nullptr;

    if (!filename.empty()) {
        char filename_try[MAXFILENAME + 16] = "";

        strncpy(filename_try, filename.c_str(), MAXFILENAME - 1);
        // strncpy doesnt append the trailing nullptr, of the string is too
        // long.
        filename_try[MAXFILENAME] = '\0';

        uf = unzOpen64(filename.c_str());
        if (uf == nullptr) {
            strcat(filename_try, ".zip");
            uf = unzOpen64(filename_try);
        }
    }

    if (uf == nullptr) {
        utility::LogError("Failed to open file {}.", filename);
    }

    // Change working directory to the extraction directory.
    if (chdir(extract_dir.c_str())) {
        utility::LogError("Error extracting to {}", extract_dir);
    }

    // ExtractFromZIP supports password. Can be exposed if required in future.
    const std::string password = "";
    ExtractAll(uf, password);

    unzClose(uf);
}

}  // namespace data
}  // namespace open3d
