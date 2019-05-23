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

#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>

#include "Open3D/Utility/FileSystem.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Get the file extension and convert to lower case.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileExtensionInLowerCase) {
    string path;
    string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // just a dot
    path = "fileName.";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // dot before the /
    path = "test/utility::filesystem.EXT/fileName";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // space in extension
    path = "test/test_dir/fileName. EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ(" ext", result);
}

// ----------------------------------------------------------------------------
// Should return the file name only, without extension.
// What it actually does is return the full path without the extension.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutExtension) {
    string path;
    string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("fileName", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName.EXT", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName.", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName ", result);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutDirectory) {
    string path;
    string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT.EXT", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName..EXT", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName .EXT", result);
}

// ----------------------------------------------------------------------------
// Get parent directory, terminated in '/'.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileParentDirectory) {
    string path;
    string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);
}

// ----------------------------------------------------------------------------
// Add '/' at the end of the input path, if missing.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetRegularizedDirectoryName) {
    string path;
    string result;

    path = "";
    result = utility::filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("/", result);

    path = "test/test_dir";
    result = utility::filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/test_dir/", result);

    path = "test/test_dir/";
    result = utility::filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/test_dir/", result);
}

// ----------------------------------------------------------------------------
// Get/Change the working directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ChangeWorkingDirectory) {
    string path = "test";

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    string cwd = utility::filesystem::GetWorkingDirectory();

    EXPECT_EQ(path, utility::filesystem::GetFileNameWithoutDirectory(cwd));

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Check if a path exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, DirectoryExists) {
    string path = "test/test_dir";

    bool status;

    // path doesn't exist yet
    status = utility::filesystem::DirectoryExists(path);
    EXPECT_FALSE(status);

    // create the path
    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // path exists
    status = utility::filesystem::DirectoryExists(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a directory.
// Return true if the directory was created.
// Return false otherwise. This could mean that the directory already exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectory) {
    string path = "test";

    bool status;

    status = utility::filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::MakeDirectory(path);
    EXPECT_FALSE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a hierarchy of directories. Equivalent to 'mkdir -p ...'.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectoryHierarchy) {
    string path = "test/test_dir";

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Note: DeleteDirectory can delete one dir at a time.
// ----------------------------------------------------------------------------
TEST(FileSystem, DeleteDirectory) {
    string path = "test";

    bool status;

    status = utility::filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory(path);
    EXPECT_FALSE(status);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, File_Exists_Remove) {
    string path = "test/test_dir";
    string fileName = "fileName.ext";

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    creat(fileName.c_str(), 0);

    status = utility::filesystem::FileExists(fileName);
    EXPECT_TRUE(status);

    status = utility::filesystem::RemoveFile(fileName);
    EXPECT_TRUE(status);

    status = utility::filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// List all files in the specified directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ListFilesInDirectory) {
    string path = "test/test_dir";
    vector<string> fileNames = {"fileName00.ext", "fileName01.ext",
                                "fileName02.ext", "fileName03.ext"};

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    for (size_t i = 0; i < fileNames.size(); i++)
        creat(fileNames[i].c_str(), 0);

    vector<string> list;
    status = utility::filesystem::ListFilesInDirectory(".", list);
    EXPECT_TRUE(status);

    sort(list.begin(), list.end());

    for (size_t i = 0; i < fileNames.size(); i++) {
        EXPECT_EQ(fileNames[i],
                  utility::filesystem::GetFileNameWithoutDirectory(list[i]));
    }

    // clean-up
    for (size_t i = 0; i < fileNames.size(); i++) {
        status = utility::filesystem::RemoveFile(fileNames[i]);
        EXPECT_TRUE(status);
    }

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// List all files of a specific extension in the specified directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ListFilesInDirectoryWithExtension) {
    string path = "test/test_dir";
    vector<string> fileNames = {"fileName00.ext0", "fileName01.ext0",
                                "fileName02.ext0", "fileName03.ext0",
                                "fileName04.ext1", "fileName05.ext1",
                                "fileName06.ext1", "fileName07.ext1"};

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    for (size_t i = 0; i < fileNames.size(); i++)
        creat(fileNames[i].c_str(), 0);

    vector<string> list;
    status = utility::filesystem::ListFilesInDirectory(".", list);
    EXPECT_TRUE(status);

    sort(list.begin(), list.end());

    for (size_t i = 0; i < list.size(); i++) {
        EXPECT_EQ(fileNames[i],
                  utility::filesystem::GetFileNameWithoutDirectory(list[i]));
    }

    // clean-up
    for (size_t i = 0; i < fileNames.size(); i++) {
        status = utility::filesystem::RemoveFile(fileNames[i]);
        EXPECT_TRUE(status);
    }

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}
