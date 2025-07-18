// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace open3d {
namespace utility {
namespace filesystem {

/// \brief Get the HOME directory for the user.
///
/// The home directory is determined in the following order:
/// - On Unix:
///   - $HOME
///   - /
/// - On Windows:
///   - %USERPROFILE%
///   - %HOMEDRIVE%
///   - %HOMEPATH%
///   - %HOME%
///   - C:/
///
/// This is the same logics as used in Qt.
/// - src/corelib/io/qfilesystemengine_win.cpp
/// - src/corelib/io/qfilesystemengine_unix.cpp
std::string GetHomeDirectory();

std::string GetFileExtensionInLowerCase(const std::string &filename);

std::string GetFileNameWithoutExtension(const std::string &filename);

std::string GetFileNameWithoutDirectory(const std::string &filename);

std::string GetFileParentDirectory(const std::string &filename);

std::string GetRegularizedDirectoryName(const std::string &directory);

std::string GetWorkingDirectory();

std::vector<std::string> GetPathComponents(const std::string &path);

std::string GetTempDirectoryPath();

bool ChangeWorkingDirectory(const std::string &directory);

bool DirectoryExists(const std::string &directory);

// Return true if the directory is present and empty. Return false if the
// directory is present but not empty. Throw an exception if the directory is
// not present.
bool DirectoryIsEmpty(const std::string &directory);

bool MakeDirectory(const std::string &directory);

bool MakeDirectoryHierarchy(const std::string &directory);

bool DeleteDirectory(const std::string &directory);

bool FileExists(const std::string &filename);

bool Copy(const std::string &src_path, const std::string &dst_path);

bool RemoveFile(const std::string &filename);

bool ListDirectory(const std::string &directory,
                   std::vector<std::string> &subdirs,
                   std::vector<std::string> &filenames);

bool ListFilesInDirectory(const std::string &directory,
                          std::vector<std::string> &filenames);

bool ListFilesInDirectoryWithExtension(const std::string &directory,
                                       const std::string &extname,
                                       std::vector<std::string> &filenames);

std::vector<std::string> FindFilesRecursively(
        const std::string &directory,
        std::function<bool(const std::string &)> is_match);

// wrapper for fopen that enables unicode paths on Windows
FILE *FOpen(const std::string &filename, const std::string &mode);
std::string GetIOErrorString(const int errnoVal);
bool FReadToBuffer(const std::string &path,
                   std::vector<char> &bytes,
                   std::string *errorStr);

std::string JoinPath(const std::vector<std::string> &path_components);

std::string JoinPath(const std::string &path_component1,
                     const std::string &path_component2);

std::string AddIfExist(const std::string &path,
                       const std::vector<std::string> &folder_names);

/// RAII Wrapper for C FILE*
/// Throws exceptions in situations where the caller is not usually going to
/// have proper handling code:
/// - using an unopened CFile
/// - errors and ferror from underlying calls (fclose, ftell, fseek, fread,
///   fwrite, fgetpos, fsetpos)
/// - reading a line that is too long, caller is unlikely to have proper code to
///   handle a partial next line
/// If you would like to handle any of these issues by not having your code
/// throw, use try/catch (const std::exception &e) { ... }
class CFile {
public:
    /// The destructor closes the file automatically.
    ~CFile();

    /// Open a file.
    bool Open(const std::string &filename, const std::string &mode);

    /// Returns the last encountered error for this file.
    std::string GetError();

    /// Close the file.
    void Close();

    /// Returns current position in the file (ftell).
    int64_t CurPos();

    /// Returns the file size in bytes.
    int64_t GetFileSize();

    /// Returns the number of lines in the file.
    int64_t GetNumLines();

    /// Throws if we hit buffer maximum. In most cases, calling code is only
    /// capable of processing a complete line, if it receives a partial line it
    /// will probably fail and it is very likely to fail/corrupt on the next
    /// call that receives the remainder of the line.
    const char *ReadLine();

    /// Read data to a buffer.
    /// \param data The data buffer to be written into.
    /// \param num_elems Number of elements to be read. The byte size of the
    /// element is determined by the size of buffer type.
    template <class T>
    size_t ReadData(T *data, size_t num_elems) {
        return ReadData(data, sizeof(T), num_elems);
    }

    /// Read data to a buffer.
    /// \param data The data buffer to be written into.
    /// \param elem_size Element size in bytes.
    /// \param num_elems Number of elements to read.
    size_t ReadData(void *data, size_t elem_size, size_t num_elems);

    /// Returns the underlying C FILE pointer.
    FILE *GetFILE() { return file_; }

private:
    FILE *file_ = nullptr;
    int error_code_ = 0;
    std::vector<char> line_buffer_;
};

}  // namespace filesystem
}  // namespace utility
}  // namespace open3d
