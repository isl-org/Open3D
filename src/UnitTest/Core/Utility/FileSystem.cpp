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

#include "Utility/UnitTest.h"

#include "Core/Utility/FileSystem.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Get the file extension and convert to lower case.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileExtensionInLowerCase)
{
    string path;
    string result;

    // empty
    path = "";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // just a dot
    path = "fileName.";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    path = "test/filesystem/fileName.EXT";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // no extension
    path = "test/filesystem/fileName";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // multiple extensions
    path = "test/filesystem/fileName.EXT.EXT";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // multiple dots
    path = "test/filesystem/fileName..EXT";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // dot before the /
    path = "test/filesystem.EXT/fileName";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // space in file name
    path = "test/filesystem/fileName .EXT";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // space in extension
    path = "test/filesystem/fileName. EXT";
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ(" ext", result);
}

// ----------------------------------------------------------------------------
// Should return the file name only, without extension.
// What it actually does is return the full path without the extension.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutExtension)
{
    string path;
    string result;

    // empty
    path = "";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("fileName", result);

    path = "test/filesystem/fileName.EXT";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/filesystem/fileName", result);

    // no extension
    path = "test/filesystem/fileName";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/filesystem/fileName", result);

    // multiple extensions
    path = "test/filesystem/fileName.EXT.EXT";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/filesystem/fileName.EXT", result);

    // multiple dots
    path = "test/filesystem/fileName..EXT";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/filesystem/fileName.", result);

    // space in file name
    path = "test/filesystem/fileName .EXT";
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/filesystem/fileName ", result);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutDirectory)
{
    string path;
    string result;

    // empty
    path = "";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT", result);

    path = "test/filesystem/fileName.EXT";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT", result);

    // no extension
    path = "test/filesystem/fileName";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName", result);

    // multiple extensions
    path = "test/filesystem/fileName.EXT.EXT";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT.EXT", result);

    // multiple dots
    path = "test/filesystem/fileName..EXT";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName..EXT", result);

    // space in file name
    path = "test/filesystem/fileName .EXT";
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName .EXT", result);
}

// ----------------------------------------------------------------------------
// Get parent directory, terminated in '/'.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileParentDirectory)
{
    string path;
    string result;

    // empty
    path = "";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("", result);

    path = "test/filesystem/fileName.EXT";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/filesystem/", result);

    // no extension
    path = "test/filesystem/fileName";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/filesystem/", result);

    // multiple extensions
    path = "test/filesystem/fileName.EXT.EXT";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/filesystem/", result);

    // multiple dots
    path = "test/filesystem/fileName..EXT";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/filesystem/", result);

    // space in file name
    path = "test/filesystem/fileName .EXT";
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/filesystem/", result);
}

// ----------------------------------------------------------------------------
// Add '/' at the end of the input path, if missing.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetRegularizedDirectoryName)
{
    string path;
    string result;

    path = "";
    result = filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("/", result);

    path = "test/filesystem";
    result = filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/filesystem/", result);

    path = "test/filesystem/";
    result = filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/filesystem/", result);
}

// ----------------------------------------------------------------------------
// Get/Change the working directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ChangeWorkingDirectory)
{
    string path = "test";

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    string cwd = filesystem::GetWorkingDirectory();

    EXPECT_EQ(path, filesystem::GetFileNameWithoutDirectory(cwd));

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Check if a path exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, DirectoryExists)
{
    string path = "test/filesystem";

    bool status;

    // path doesn't exist yet
    status = filesystem::DirectoryExists(path);
    EXPECT_FALSE(status);

    // create the path
    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // path exists
    status = filesystem::DirectoryExists(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a directory.
// Return true if the directory was created.
// Return false otherwise. This could mean that the directory already exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectory)
{
    string path = "test";

    bool status;

    status = filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::MakeDirectory(path);
    EXPECT_FALSE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a hierarchy of directories. Equivalent to 'mkdir -p ...'.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectoryHierarchy)
{
    string path = "test/filesystem";

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Note: DeleteDirectory can delete one dir at a time.
// ----------------------------------------------------------------------------
TEST(FileSystem, DeleteDirectory)
{
    string path = "test";

    bool status;

    status = filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory(path);
    EXPECT_FALSE(status);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, File_Exists_Remove)
{
    string path = "test/filesystem";
    string fileName = "fileName.ext";

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    creat(fileName.c_str(), 0);

    status = filesystem::FileExists(fileName);
    EXPECT_TRUE(status);

    status = filesystem::RemoveFile(fileName);
    EXPECT_TRUE(status);

    status = filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// List all files in the specified directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ListFilesInDirectory)
{
    string path = "test/filesystem";
    vector<string> fileNames = { "fileName00.ext",
                                 "fileName01.ext",
                                 "fileName02.ext",
                                 "fileName03.ext" };

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    for (size_t i = 0; i < fileNames.size(); i++)
        creat(fileNames[i].c_str(), 0);

    vector<string> list;
    status = filesystem::ListFilesInDirectory(".", list);
    EXPECT_TRUE(status);

    sort(list.begin(), list.end());

    for (size_t i = 0; i < fileNames.size(); i++)
    {
        EXPECT_EQ(fileNames[i],
                  filesystem::GetFileNameWithoutDirectory(list[i]));
    }

    // clean-up
    for (size_t i = 0; i < fileNames.size(); i++)
    {
        status = filesystem::RemoveFile(fileNames[i]);
        EXPECT_TRUE(status);
    }

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// List all files of a specific extension in the specified directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ListFilesInDirectoryWithExtension)
{
    string path = "test/filesystem";
    vector<string> fileNames = { "fileName00.ext0",
                                 "fileName01.ext0",
                                 "fileName02.ext0",
                                 "fileName03.ext0",
                                 "fileName04.ext1",
                                 "fileName05.ext1",
                                 "fileName06.ext1",
                                 "fileName07.ext1" };

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    for (size_t i = 0; i < fileNames.size(); i++)
        creat(fileNames[i].c_str(), 0);

    vector<string> list;
    status = filesystem::ListFilesInDirectory(".", list);
    EXPECT_TRUE(status);

    sort(list.begin(), list.end());

    for (size_t i = 0; i < list.size(); i++)
    {
        EXPECT_EQ(fileNames[i],
                  filesystem::GetFileNameWithoutDirectory(list[i]));
    }

    // clean-up
    for (size_t i = 0; i < fileNames.size(); i++)
    {
        status = filesystem::RemoveFile(fileNames[i]);
        EXPECT_TRUE(status);
    }

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}
