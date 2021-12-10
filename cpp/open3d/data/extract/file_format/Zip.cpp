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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>

#include "open3d/data/extract/Extract.h"
#include "open3d/data/extract/minizip/unzip.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

#ifdef WIN32
#define USEWIN32IOAPI
#include "open3d/data/extract/minizip/iowin32.h"
#endif

#ifdef unix
#include <unistd.h>
#include <utime.h>
#else
#include <direct.h>
#include <io.h>
#endif

#define CASESENSITIVITY (0)
#define WRITEBUFFERSIZE (8192)
#define MAXFILENAME (256)

namespace open3d {
namespace data {

static int ExtractCurrentFile(unzFile uf) {
    char filename_inzip[256];
    char *filename_withoutpath;
    char *p;
    int err = UNZ_OK;
    FILE *fout = NULL;
    void *buf;
    uInt size_buf;
    const char *password = NULL;
    // int opt_do_list = 0;
    // int opt_do_extract = 1;
    int opt_do_extract_withoutpath = 0;
    // int opt_overwrite = 0;
    // int opt_extractdir = 0;

    unz_file_info file_info;
    // uLong ratio = 0;
    err = unzGetCurrentFileInfo(uf, &file_info, filename_inzip,
                                sizeof(filename_inzip), NULL, 0, NULL, 0);

    if (err != UNZ_OK) {
        printf("error %d with zipfile in unzGetCurrentFileInfo\n", err);
        return err;
    }

    size_buf = WRITEBUFFERSIZE;
    buf = (void *)malloc(size_buf);
    if (buf == NULL) {
        printf("Error allocating memory\n");
        return UNZ_INTERNALERROR;
    }

    p = filename_withoutpath = filename_inzip;
    while ((*p) != '\0') {
        if (((*p) == '/') || ((*p) == '\\')) filename_withoutpath = p + 1;
        p++;
    }

    if ((*filename_withoutpath) == '\0') {
        if (opt_do_extract_withoutpath == 0) {
            utility::LogDebug("Creating directory: {}", filename_inzip);
            // mymkdir(filename_inzip);
            utility::filesystem::MakeDirectoryHierarchy(
                    std::string(filename_inzip));
        }
    } else {
        const char *write_filename;
        // int skip = 0;

        if (opt_do_extract_withoutpath == 0)
            write_filename = filename_inzip;
        else
            write_filename = filename_withoutpath;

        err = unzOpenCurrentFilePassword(uf, password);
        if (err != UNZ_OK) {
            printf("error %d with zipfile in unzOpenCurrentFilePassword\n",
                   err);
        }

        if (err == UNZ_OK) {
            fout = fopen(write_filename, "wb");

            /* some zipfile don't contain directory alone before file */
            if ((fout == NULL) && (opt_do_extract_withoutpath == 0) &&
                (filename_withoutpath != (char *)filename_inzip)) {
                char c = *(filename_withoutpath - 1);
                *(filename_withoutpath - 1) = '\0';

                // makedir(write_filename);
                utility::filesystem::MakeDirectoryHierarchy(
                        std::string(filename_inzip));

                *(filename_withoutpath - 1) = c;
                fout = fopen(write_filename, "wb");
            }

            if (fout == NULL) {
                printf("error opening %s\n", write_filename);
            }
        }

        if (fout != NULL) {
            utility::LogDebug(" Extracting: {}", write_filename);

            do {
                err = unzReadCurrentFile(uf, buf, size_buf);
                if (err < 0) {
                    printf("error %d with zipfile in unzReadCurrentFile\n",
                           err);
                    break;
                }
                if (err > 0)
                    if (fwrite(buf, err, 1, fout) != 1) {
                        printf("error in writing extracted file\n");
                        err = UNZ_ERRNO;
                        break;
                    }
            } while (err > 0);
            if (fout) fclose(fout);

            // if (err == 0)
            //     change_file_date(write_filename, file_info.dosDate,
            //                      file_info.tmu_date);
        }

        if (err == UNZ_OK) {
            err = unzCloseCurrentFile(uf);
            if (err != UNZ_OK) {
                printf("error %d with zipfile in unzCloseCurrentFile\n", err);
            }
        } else
            unzCloseCurrentFile(uf); /* don't lose the error */
    }

    free(buf);
    return err;
}

static bool ExtractAll(unzFile uf,
                       const std::string &password,
                       const bool always_overwrite) {
    uLong i;
    unz_global_info gi;
    int err;

    err = unzGetGlobalInfo(uf, &gi);
    if (err != UNZ_OK)
        printf("error %d with zipfile in unzGetGlobalInfo \n", err);

    for (i = 0; i < gi.number_entry; i++) {
        if (ExtractCurrentFile(uf) != UNZ_OK) break;

        if ((i + 1) < gi.number_entry) {
            err = unzGoToNextFile(uf);
            if (err != UNZ_OK) {
                utility::LogWarning("Error {} with zipfile in unzGoToNextFile",
                                    err);
                break;
            }
        }
    }

    return true;
}

bool ExtractFromZIP(const std::string &filename,
                    const std::string &extract_dir,
                    const std::string &password,
                    const bool always_overwrite,
                    const bool print_progress) {
    // const char* filename_to_extract = NULL;

    char filename_try[MAXFILENAME + 16] = "";

    // int i;
    // int opt_do_list = 0;
    // int opt_do_extract_withoutpath = 0;
    // int opt_overwrite = 1;

    unzFile uf = NULL;

    if (!filename.empty()) {
#ifdef USEWIN32IOAPI
        zlib_filefunc_def ffunc;
#endif

        strncpy(filename_try, filename.c_str(), MAXFILENAME - 1);
        /* strncpy doesnt append the trailing NULL, of the string is too long.
         */
        filename_try[MAXFILENAME] = '\0';

#ifdef USEWIN32IOAPI
        fill_win32_filefunc(&ffunc);
        uf = unzOpen2(zipfilename, &ffunc);
#else
        uf = unzOpen(filename.c_str());
#endif
        if (uf == NULL) {
            strcat(filename_try, ".zip");
#ifdef USEWIN32IOAPI
            uf = unzOpen2(filename_try, &ffunc);
#else
            uf = unzOpen(filename_try);
#endif
        }
    }

    if (uf == NULL) {
        utility::LogWarning("Failed to open file {}.", filename);
        return false;
    }

    // Change working directory to the extraction directory.
    if (chdir(extract_dir.c_str())) {
        utility::LogError("Error extracting to {}", extract_dir);
    }

    int success = ExtractAll(uf, password, always_overwrite);

    unzCloseCurrentFile(uf);
    return success;
}

}  // namespace data
}  // namespace open3d
