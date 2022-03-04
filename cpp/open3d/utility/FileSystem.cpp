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

#include "open3d/utility/FileSystem.h"

#include <fcntl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#ifdef WIN32
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

#ifdef WIN32
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif
#ifdef __APPLE__
// CMAKE_OSX_DEPLOYMENT_TARGET "10.15" or newer
#define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#endif
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {
namespace filesystem {

static std::string GetEnvVar(const std::string &env_var) {
    if (const char *env_p = std::getenv(env_var.c_str())) {
        return std::string(env_p);
    } else {
        return "";
    }
}

std::string GetHomeDirectory() {
    std::string home_dir = "";
#ifdef WINDOWS
    // %USERPROFILE%
    // %HOMEDRIVE%
    // %HOMEPATH%
    // %HOME%
    // C:/
    home_dir = GetEnvVar("USERPROFILE");
    if (home_dir.empty() || !DirectoryExists(home_dir)) {
        home_dir = GetEnvVar("HOMEDRIVE");
        if (home_dir.empty() || !DirectoryExists(home_dir)) {
            home_dir = GetEnvVar("HOMEPATH");
            if (home_dir.empty() || !DirectoryExists(home_dir)) {
                home_dir = GetEnvVar("HOME");
                if (home_dir.empty() || !DirectoryExists(home_dir)) {
                    home_dir = "C:/";
                }
            }
        }
    }
#else
    // $HOME
    // /
    home_dir = GetEnvVar("HOME");
    if (home_dir.empty() || !DirectoryExists(home_dir)) {
        home_dir = "/";
    }
#endif
    return home_dir;
}

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
    if (directory.empty()) {
        return "/";
    } else if (directory.back() != '/' && directory.back() != '\\') {
        return directory + "/";
    } else {
        return directory;
    }
}

std::string GetWorkingDirectory() {
    char buff[PATH_MAX + 1];
    auto ignored = getcwd(buff, PATH_MAX + 1);
    (void)ignored;
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
    if (isRelative) {
        auto cwd = utility::filesystem::GetWorkingDirectory();
        auto cwdComponents = SplitByPathSeparators(cwd);
        components.insert(components.end(), cwdComponents.begin(),
                          cwdComponents.end());
        if (cwd[0] != '/') {
            isWindowsPath = true;
        }
    } else {
        // absolute path, don't need any prefix
    }
    if (!isWindowsPath) {
        components.insert(components.begin(), "/");
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
    return fs::is_directory(directory);
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
        if (!DirectoryExists(subdir)) {
            if (!MakeDirectory(subdir)) {
                return false;
            }
        }
        curr_pos = full_path.find_first_of("/\\", curr_pos + 1);
    }
    return true;
}

bool DeleteDirectory(const std::string &directory) {
    std::error_code error;
    if (fs::remove_all(directory, error) == static_cast<std::uintmax_t>(-1)) {
        utility::LogWarning("Failed to remove directory {}: {}.", directory,
                            error.message());
        return false;
    }
    return true;
}

bool FileExists(const std::string &filename) {
    return fs::exists(filename) && fs::is_regular_file(filename);
}

// TODO: this is not a good name. Currently FileSystem.cpp includes windows.h
// and "CopyFile" will be expanded to "CopyFileA" on Windows. This will be
// resolved when we switch to C++17's std::filesystem.
bool Copy(const std::string &src_path, const std::string &dst_path) {
    try {
        fs::copy(src_path, dst_path,
                 fs::copy_options::recursive |
                         fs::copy_options::overwrite_existing);
    } catch (std::exception &e) {
        utility::LogWarning("Failed to copy {} to {}. Exception: {}.", src_path,
                            dst_path, e.what());
        return false;
    }
    return true;
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
    if (!ListFilesInDirectory(directory, all_files)) {
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

std::vector<std::string> FindFilesRecursively(
        const std::string &directory,
        std::function<bool(const std::string &)> is_match) {
    std::vector<std::string> matches;

    std::vector<std::string> subdirs;
    std::vector<std::string> files;
    ListDirectory(directory, subdirs, files);  // results are paths
    for (auto &f : files) {
        if (is_match(f)) {
            matches.push_back(f);
        }
    }
    for (auto &d : subdirs) {
        auto submatches = FindFilesRecursively(d, is_match);
        if (!submatches.empty()) {
            matches.insert(matches.end(), submatches.begin(), submatches.end());
        }
    }

    return matches;
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

CFile::~CFile() { Close(); }

bool CFile::Open(const std::string &filename, const std::string &mode) {
    Close();
    file_ = FOpen(filename, mode);
    if (!file_) {
        error_code_ = errno;
    }
    return bool(file_);
}

std::string CFile::GetError() { return GetIOErrorString(error_code_); }

void CFile::Close() {
    if (file_) {
        if (fclose(file_)) {
            error_code_ = errno;
            utility::LogError("fclose failed: {}", GetError());
        }
        file_ = nullptr;
    }
}

int64_t CFile::CurPos() {
    if (!file_) {
        utility::LogError("CFile::CurPos() called on a closed file");
    }
    int64_t pos = ftell(file_);
    if (pos < 0) {
        error_code_ = errno;
        utility::LogError("ftell failed: {}", GetError());
    }
    return pos;
}

int64_t CFile::GetFileSize() {
    if (!file_) {
        utility::LogError("CFile::GetFileSize() called on a closed file");
    }
    fpos_t prevpos;
    if (fgetpos(file_, &prevpos)) {
        error_code_ = errno;
        utility::LogError("fgetpos failed: {}", GetError());
    }
    if (fseek(file_, 0, SEEK_END)) {
        error_code_ = errno;
        utility::LogError("fseek failed: {}", GetError());
    }
    int64_t size = CurPos();
    if (fsetpos(file_, &prevpos)) {
        error_code_ = errno;
        utility::LogError("fsetpos failed: {}", GetError());
    }
    return size;
}

int64_t CFile::GetNumLines() {
    if (!file_) {
        utility::LogError("CFile::GetNumLines() called on a closed file");
    }
    fpos_t prevpos;
    if (fgetpos(file_, &prevpos)) {
        error_code_ = errno;
        utility::LogError("fgetpos failed: {}", GetError());
    }
    if (fseek(file_, 0, SEEK_SET)) {
        error_code_ = errno;
        utility::LogError("fseek failed: {}", GetError());
    }
    int64_t num_lines = 0;
    int c;
    while (EOF != (c = getc(file_))) {
        if (c == '\n') {
            num_lines++;
        }
    }
    if (fsetpos(file_, &prevpos)) {
        error_code_ = errno;
        utility::LogError("fsetpos failed: {}", GetError());
    }
    return num_lines;
}

const char *CFile::ReadLine() {
    if (!file_) {
        utility::LogError("CFile::ReadLine() called on a closed file");
    }
    if (line_buffer_.size() == 0) {
        line_buffer_.resize(DEFAULT_IO_BUFFER_SIZE);
    }
    if (!fgets(line_buffer_.data(), int(line_buffer_.size()), file_)) {
        if (ferror(file_)) {
            utility::LogError("CFile::ReadLine() ferror encountered");
        }
        if (!feof(file_)) {
            utility::LogError(
                    "CFile::ReadLine() fgets returned NULL, ferror is not set, "
                    "feof is not set");
        }
        return nullptr;
    }
    if (strlen(line_buffer_.data()) == line_buffer_.size() - 1) {
        // if we didn't read the whole line, chances are code using this is
        // not equipped to handle partial line on next call
        utility::LogError("CFile::ReadLine() encountered a line longer than {}",
                          line_buffer_.size() - 2);
    }
    return line_buffer_.data();
}

size_t CFile::ReadData(void *data, size_t elem_size, size_t num_elems) {
    if (!file_) {
        utility::LogError("CFile::ReadData() called on a closed file");
    }
    size_t elems = fread(data, elem_size, num_elems, file_);
    if (ferror(file_)) {
        utility::LogError("CFile::ReadData() ferror encountered");
    }
    if (elems < num_elems) {
        if (!feof(file_)) {
            utility::LogError(
                    "CFile::ReadData() fread short read, ferror not set, feof "
                    "not set");
        }
    }
    return elems;
}

}  // namespace filesystem
}  // namespace utility
}  // namespace open3d
