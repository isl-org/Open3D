// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Utility/FileSystem.h"

#include <fcntl.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#ifdef WINDOWS
#include <direct.h>
#include <dirent/dirent.h>
#include <io.h>
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace open3d {
namespace utility {
namespace filesystem {

std::string GetFileExtensionInLowerCase(const std::string &filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos >= filename.length()) return "";

    if (filename.find_first_of("/\\", dot_pos) != std::string::npos) return "";

    std::string filename_ext = filename.substr(dot_pos + 1);

    std::transform(filename_ext.begin(), filename_ext.end(),
                   filename_ext.begin(), ::tolower);

    return filename_ext;
}

std::string GetFileNameWithoutExtension(const std::string &filename) {
    size_t dot_pos = filename.find_last_of(".");

    return filename.substr(0, dot_pos);
}

std::string GetFileNameWithoutDirectory(const std::string &filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos == std::string::npos) {
        return filename;
    } else {
        return filename.substr(slash_pos + 1);
    }
}

std::string GetFileParentDirectory(const std::string &filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos == std::string::npos) {
        return "";
    } else {
        return filename.substr(0, slash_pos + 1);
    }
}

std::string GetRegularizedDirectoryName(const std::string &directory) {
    if (directory.back() != '/' && directory.back() != '\\') {
        return directory + "/";
    } else {
        return directory;
    }
}

std::string GetWorkingDirectory() {
    char buff[PATH_MAX + 1];
    getcwd(buff, PATH_MAX + 1);
    return std::string(buff);
}

std::vector<std::string> GetPathComponents(const std::string &path) {
    auto SplitByPathSeparators = [](const std::string &path) {
        std::vector<std::string> components;
        // Split path by '/' and '\'
        if (!path.empty()) {
            size_t end = 0;
            while (end < path.size()) {
                size_t start = end;
                while (end < path.size() && path[end] != '\\' &&
                       path[end] != '/') {
                    end++;
                }
                if (end > start) {
                    components.push_back(path.substr(start, end - start));
                }
                if (end < path.size()) {
                    end++;
                }
            }
        }
        return components;
    };

    auto pathComponents = SplitByPathSeparators(path.c_str());

    // Handle "/" and "" paths
    if (pathComponents.empty()) {
        if (path == "/") {
            // absolute path; the "/" component will be added
            // later, so don't do anything here
        } else {
            pathComponents.push_back(".");
        }
    }

    char firstChar = path[0];  // '/' doesn't get stored in components
    bool isRelative = (firstChar != '/');
    bool isWindowsPath = false;
    // Check for Windows full path (e.g. "d:")
    if (isRelative && pathComponents[0].size() >= 2 &&
        ((firstChar >= 'a' && firstChar <= 'z') ||
         (firstChar >= 'A' && firstChar <= 'Z')) &&
        pathComponents[0][1] == ':') {
        isRelative = false;
        isWindowsPath = true;
    }

    std::vector<std::string> components;
    if (!isWindowsPath) {
        components.push_back("/");
    }
    if (isRelative) {
        auto cwd = utility::filesystem::GetWorkingDirectory();
        auto cwdComponents = SplitByPathSeparators(cwd);
        components.insert(components.end(), cwdComponents.begin(),
                          cwdComponents.end());
    } else {
        // absolute path, don't need any prefix
    }

    for (auto &dir : pathComponents) {
        if (dir == ".") { /* ignore */
        } else if (dir == "..") {
            components.pop_back();
        } else {
            components.push_back(dir);
        }
    }

    return components;
}

bool ChangeWorkingDirectory(const std::string &directory) {
    return (chdir(directory.c_str()) == 0);
}

bool DirectoryExists(const std::string &directory) {
    struct stat info;
    if (stat(directory.c_str(), &info) == -1) return false;
    return S_ISDIR(info.st_mode);
}

bool MakeDirectory(const std::string &directory) {
#ifdef WINDOWS
    return (_mkdir(directory.c_str()) == 0);
#else
    return (mkdir(directory.c_str(), S_IRWXU) == 0);
#endif
}

bool MakeDirectoryHierarchy(const std::string &directory) {
    std::string full_path = GetRegularizedDirectoryName(directory);
    size_t curr_pos = full_path.find_first_of("/\\", 1);
    while (curr_pos != std::string::npos) {
        std::string subdir = full_path.substr(0, curr_pos + 1);
        if (DirectoryExists(subdir) == false) {
            if (MakeDirectory(subdir) == false) {
                return false;
            }
        }
        curr_pos = full_path.find_first_of("/\\", curr_pos + 1);
    }
    return true;
}

bool DeleteDirectory(const std::string &directory) {
#ifdef WINDOWS
    return (_rmdir(directory.c_str()) == 0);
#else
    return (rmdir(directory.c_str()) == 0);
#endif
}

bool FileExists(const std::string &filename) {
#ifdef WINDOWS
    struct _stat64 info;
    if (_stat64(filename.c_str(), &info) == -1) return false;
    return S_ISREG(info.st_mode);
#else
    struct stat info;
    if (stat(filename.c_str(), &info) == -1) return false;
    return S_ISREG(info.st_mode);
#endif
}

bool RemoveFile(const std::string &filename) {
    return (std::remove(filename.c_str()) == 0);
}

bool ListDirectory(const std::string &directory,
                   std::vector<std::string> &subdirs,
                   std::vector<std::string> &filenames) {
    if (directory.empty()) {
        return false;
    }
    DIR *dir;
    struct dirent *ent;
    struct stat st;
    dir = opendir(directory.c_str());
    if (!dir) {
        return false;
    }
    filenames.clear();
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        if (file_name[0] == '.') continue;
        std::string full_file_name =
                GetRegularizedDirectoryName(directory) + file_name;
        if (stat(full_file_name.c_str(), &st) == -1) continue;
        if (S_ISDIR(st.st_mode))
            subdirs.push_back(full_file_name);
        else if (S_ISREG(st.st_mode))
            filenames.push_back(full_file_name);
    }
    closedir(dir);
    return true;
}

bool ListFilesInDirectory(const std::string &directory,
                          std::vector<std::string> &filenames) {
    std::vector<std::string> subdirs;
    return ListDirectory(directory, subdirs, filenames);
}

bool ListFilesInDirectoryWithExtension(const std::string &directory,
                                       const std::string &extname,
                                       std::vector<std::string> &filenames) {
    std::vector<std::string> all_files;
    if (ListFilesInDirectory(directory, all_files) == false) {
        return false;
    }
    std::string ext_in_lower = extname;
    std::transform(ext_in_lower.begin(), ext_in_lower.end(),
                   ext_in_lower.begin(), ::tolower);
    filenames.clear();
    for (const auto &fn : all_files) {
        if (GetFileExtensionInLowerCase(fn) == ext_in_lower) {
            filenames.push_back(fn);
        }
    }
    return true;
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

static std::string GetIOErrorString(const int errnoVal) {
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

}  // namespace filesystem
}  // namespace utility
}  // namespace open3d
