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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WIN32
#include <windows.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif
#ifdef __APPLE__
#define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#endif
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

std::string GetIOErrorString(const int errnoVal) {
    switch (errnoVal) {
        case EPERM:
            return "Operation not permitted";
        case EACCES:
            return "Access denied";
        // Error below could be EWOULDBLOCK on Linux
        case EAGAIN:
            return "Resource unavailable, try again";
#if !defined(WIN32)
        case EDQUOT:
            return "Over quota";
#endif
        case EEXIST:
            return "File already exists";
        case EFAULT:
            return "Bad filename pointer";
        case EINTR:
            return "open() interrupted by a signal";
        case EIO:
            return "I/O error";
        case ELOOP:
            return "Too many symlinks, could be a loop";
        case EMFILE:
            return "Process is out of file descriptors";
        case ENAMETOOLONG:
            return "Filename is too long";
        case ENFILE:
            return "File system table is full";
        case ENOENT:
            return "No such file or directory";
        case ENOSPC:
            return "No space available to create file";
        case ENOTDIR:
            return "Bad path";
        case EOVERFLOW:
            return "File is too big";
        case EROFS:
            return "Can't modify file on read-only filesystem";
#if EWOULDBLOCK != EAGAIN
        case EWOULDBLOCK:
            return "Operation would block calling process";
#endif
        default: {
            std::stringstream s;
            s << "IO error " << errnoVal << " (see sys/errno.h)";
            return s.str();
        }
    }
}

FILE *FOpen(const std::string &filename, const std::string &mode) {
    FILE *fp;
#ifndef _WIN32
    fp = fopen(filename.c_str(), mode.c_str());
#else
    std::wstring filename_w;
    filename_w.resize(filename.size());
    int newSize = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(),
                                      static_cast<int>(filename.length()),
                                      const_cast<wchar_t *>(filename_w.c_str()),
                                      static_cast<int>(filename.length()));
    filename_w.resize(newSize);
    std::wstring mode_w(mode.begin(), mode.end());
    fp = _wfopen(filename_w.c_str(), mode_w.c_str());
#endif
    return fp;
}

bool FReadToBuffer(const std::string &path,
                   std::vector<char> &bytes,
                   std::string *errorStr) {
    bytes.clear();
    if (errorStr) {
        errorStr->clear();
    }

    FILE *file = FOpen(path.c_str(), "rb");
    if (!file) {
        if (errorStr) {
            *errorStr = GetIOErrorString(errno);
        }

        return false;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        // We ignore that fseek will block our process
        if (errno && errno != EWOULDBLOCK) {
            if (errorStr) {
                *errorStr = GetIOErrorString(errno);
            }

            fclose(file);
            return false;
        }
    }

    const size_t filesize = ftell(file);
    rewind(file);  // reset file pointer back to beginning

    bytes.resize(filesize);
    const size_t result = fread(bytes.data(), 1, filesize, file);

    if (result != filesize) {
        if (errorStr) {
            *errorStr = GetIOErrorString(errno);
        }

        fclose(file);
        return false;
    }

    fclose(file);
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <input_file> <compiled_resources_folder>";
        return 0;
    }

    std::string input_file = argv[1];
    std::string var_name = fs::path(input_file).filename().string();
    var_name.replace(var_name.find('.'), 1, "_");
    if (var_name.find('-') != std::string::npos) {
        var_name.replace(var_name.find('-'), 1, "_");
    }

    std::string output_cpp_file =
            (fs::path(argv[2]) / (var_name + ".cpp")).string();
    std::string output_h_file =
            (fs::path(argv[2]) / (var_name + ".h")).string();

    std::vector<char> resource_data;
    std::string error_str;

    if (FReadToBuffer(input_file, resource_data, &error_str)) {
        std::ofstream cpp_out;
        std::ofstream h_out;

        cpp_out.open(output_cpp_file, std::ios::trunc);
        h_out.open(output_h_file, std::ios::trunc);

        std::stringstream byte_data;

        h_out << "#include <cstddef>\n"
              << "extern const char " << var_name << "_array[];\n"
              << "extern size_t " << var_name << "_count;" << std::endl;

        cpp_out << "#include \"" << fs::path(output_h_file).filename().string()
                << "\"\n"
                << "const char " << var_name << "_array[] = {\n"
                << "    ";

        for (auto byte : resource_data) {
            cpp_out << (int)byte << ",";
        }

        cpp_out << "\n};\n"
                << "size_t " << var_name << "_count =\n"
                << "        sizeof(" << var_name << "_array)/sizeof(char);"
                << std::endl;

        cpp_out.close();
        h_out.close();
    } else {
        std::cout << "Error loading file: " << error_str << std::endl;
    }
    return 0;
}
